import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch.autograd import Variable
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

def dot_attention(code, seq, mask):
	attn_score = torch.matmul(code, seq.transpose(1, 2))  # B, M, L
	attn_score = torch.tanh(attn_score)  # B, M, L
	if mask is not None:
		attn_score = attn_score + (1.0 - mask.float()[:, None, :]) * -1e5
	attn_prob = torch.softmax(attn_score, dim=-1)
	return attn_prob


def cosine(code, seq, mask):
	prod = torch.matmul(code, seq.transpose(1, 2))  # B, M, L
	norm_code = torch.norm(code, p=2, dim=-1)
	norm_seq = torch.norm(seq, p=2, dim=-1)
	cos = prod / (norm_code[:, :, None] * norm_seq[:, None, :])
	if mask is not None:
		cos = cos + (1.0 - mask.float()[:, None, :]) * -1e5
	cos = torch.softmax(cos, dim=-1)
	return cos

class Poly(fewshot_re_kit.framework.FewShotREModel):

	def __init__(self, sentence_encoder, hidden_size=230):
		fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
		self.hidden_size = hidden_size

		self.M = 16
		self.gnn_hidden = 768
		self.poly_code = nn.Parameter(torch.randn(self.M, hidden_size))
		self.dim_reduce = nn.Linear(self.hidden_size, self.gnn_hidden // self.M)
		layer = nn.TransformerEncoderLayer(d_model=self.gnn_hidden, nhead=self.M,
										   dim_feedforward=self.gnn_hidden, dropout=0.2)
		self.gnn = nn.TransformerEncoder(encoder_layer=layer, num_layers=2)
		self.sim = dot_attention
		# self.sim = cosine

	def __dist__(self, x, y, dim):
		return (torch.pow(x - y, 2)).sum(dim)

	def __batch_dist__(self, S, Q, dim):
		return self.__dist__(S, Q, dim)

	def poly_att(self, code, s, mask, sim_fn):
		if code.dim() == 2:
			code = code.unsqueeze(0)
		sim = sim_fn(code, s, mask)  # B, M, L
		out = sim @ s  # B, M, D
		return out

	def AoA(self, src, tgt, src_mask, tgt_mask, sim_fn=dot_attention):
		src_code = self.poly_att(self.poly_code, src, src_mask, sim_fn)  # B, M, D
		tgt_code = self.poly_att(src_code, tgt, tgt_mask, sim_fn)  # B, M, D
		return src_code, tgt_code

	def fuse(self, m1, m2, dim):
		return torch.cat([m1, m2, torch.abs(m1 - m2), m1 * m2], dim)

	def loss(self, logits, label):
		'''
		logits: Logits with the size (..., class_num)
		label: Label with whatever size.
		return: [Loss] (A single value)
		'''
		logits, J_incon = logits
		N = logits.size(-1)
		return self.cost(logits.view(-1, N), label.view(-1)) + J_incon

	def forward(self, support, query, N, K, NQ):
		'''
		support: Inputs of the support set.
		query: Inputs of the query set.
		N: Num of classes
		K: Num of instances for each class in the support set
		Q: Num of instances in the query set
		'''
		support_mask = support['mask']
		query_mask = query['mask']
		support_length = support_mask.sum(1).max().item()
		query_length = query_mask.sum(1).max().item()
		support_seq = self.sentence_encoder(support, pool=False)[:, :support_length]  # (B*N*K, LS, D)
		query_seq = self.sentence_encoder(query, pool=False)[:, :query_length]  # (B*NQ, LQ, D)
		support_mask = support_mask[:, :support_length]
		query_mask = query_mask[:, :query_length]

		query_code = self.poly_att(self.poly_code, query_seq, query_mask, self.sim)  # B*NQ, M, D
		query_code = self.dim_reduce(query_code).contiguous().view(-1, NQ, self.gnn_hidden)  # B, NQ, GD

		support_code = self.poly_att(self.poly_code, support_seq, support_mask, self.sim)  # B, N*K,GD
		support_code = self.dim_reduce(support_code).contiguous().view(-1, N*K, self.gnn_hidden)

		w = torch.cat([support_code, query_code], 1)  # B, N*K+NQ, GD
		w = self.gnn(src=w.transpose(0, 1)).transpose(0, 1)  # B, N*K + NQ, GD
		w_q = w[:, N * K:]
		w = w[:, :N * K]

		D = self.gnn_hidden
		w = w.contiguous().view(-1, K, D)  # B*N, K, D
		proto = w.mean(1)   # prototype  B*N, D

		J_incon = torch.sum((proto.unsqueeze(1) * w), 2)
		J_incon = 1 - (J_incon / (torch.norm(proto.unsqueeze(1), dim=-1) * torch.norm(w, dim=-1)))
		J_incon = J_incon.mean()
		proto = proto.view(-1, N, D).unsqueeze(1).expand(-1, NQ, N, D)  # B, total_Q, N, D
		w_q = w_q.contiguous().view(-1, NQ, 1, D)
		logits = torch.sum((w_q * proto), dim=3).view(-1, N)

		_, pred = torch.max(logits, 1)
		return (logits, J_incon), pred




