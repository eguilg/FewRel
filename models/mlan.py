import sys

sys.path.append('..')
import fewshot_re_kit
import numpy as np
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
# from .attention_layer import SelfAttention

def position_encoding_init(n_position, emb_dim):
	''' Init the sinusoid position encoding table '''
	# keep dim 0 for padding token position encoding zero vector
	position_enc = np.array([
		[pos / np.power(10000, 2.0 * (j // 2) / emb_dim) for j in range(emb_dim)]
		for pos in range(n_position)])
	position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
	position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim
	return torch.from_numpy(position_enc).float()


class MLAN(fewshot_re_kit.framework.FewShotREModel):
	def __init__(self, sentence_encoder, N, hidden_size=230):
		fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
		self.hidden_size = hidden_size
		pos_embs = position_encoding_init(sentence_encoder.max_length, hidden_size)
		self.positional_embeddings = nn.Embedding.from_pretrained(pos_embs, freeze=True)
		# for sentence level cross attention
		layer = nn.TransformerEncoderLayer(d_model=hidden_size+N, nhead=5, dim_feedforward=hidden_size, dropout=0.2)
		self.transformer = nn.TransformerEncoder(encoder_layer=layer, num_layers=1)
		# layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=5, dim_feedforward=hidden_size, dropout=0.2)
		self.gnn = nn.TransformerEncoder(encoder_layer=layer, num_layers=2)

		self.fc = nn.Linear(8*hidden_size+8*N, hidden_size, bias=True)
		emb_w = torch.eye(N)
		self.label_embeddings = nn.Embedding.from_pretrained(emb_w, freeze=True)


	def fuse(self, m1, m2, dim):
		return torch.cat([m1, m2, torch.abs(m1 - m2), m1 * m2], dim)

	def seq_pooling(self, seq, mask):
		mask = mask.float()
		mean = seq.sum(1) / mask.sum(1, keepdim=True)
		max = torch.max((seq - (1.0 - mask[:, :, None]) * 10000), dim=1)[0]
		vec = torch.cat([mean, max], dim=1)
		return vec

	def forward(self, support, query, N, K, Q):
		'''
		support: Inputs of the support set.
		query: Inputs of the query set.
		N: Num of classes
		K: Num of instances for each class in the support set
		Q: Num of instances for each class in the query set
		'''
		support_mask = support['mask']  # (B*N*K, L)
		query_mask = query['mask']  # (B*NQ, L)

		# get support and query lengths
		SL = support_mask.sum(1).max().item()  # support batch max length
		QL = query_mask.sum(1).max().item()  # query batch max length
		D = self.hidden_size  # hidden size

		# get encodings
		support = self.sentence_encoder(support, pool=False)  # (B * N * K, L, D)
		query = self.sentence_encoder(query, pool=False)  # （B * NQ, L, D）

		# cut paddings
		support = support[:, :SL]  # cut pads
		support_mask = support_mask[:, :SL]
		query = query[:, :QL]  # cut pads
		query_mask = query_mask[:, :QL]

		# get batch size and NQ
		support = support.contiguous().view(-1, N * K, SL, D)  # (B, N * K, SL, D)
		B = support.shape[0]  # batch size
		query = query.contiguous().view(B, -1, QL, D)  # (B, NQ, QL, D)
		NQ = query.shape[1]  # num of instances for each batch in query set



		########### sentence level cross attention ##########
		# all query in 1 line
		query = query.contiguous().view(B, NQ * QL, D)  # (B, NQ*QL, D)
		query_mask = query_mask.contiguous().view(B, NQ * QL)  # (B, NQ*QL)
		# all support in 1 line
		support = support.contiguous().view(B, N * K * SL, D)  # (B, N*K*SL, D)
		support_mask = support_mask.contiguous().view(B, N*K*SL)

		support_idxs = torch.arange(0, N, device=support.device)
		support_lb_emb = self.label_embeddings(support_idxs)  # N, N
		query_lb_emb = torch.tensor([1.0/N]*N, device=query.device)  # N
		support_lb = support_lb_emb[None, :, None, :].expand(B, -1, K * SL, -1).contiguous().view(B, N*K*SL, -1)
		query_lb = query_lb_emb[None, None, :].expand(B, NQ*QL, -1)
		support = torch.cat([support_lb, support], dim=2)  # B, N*K*SL, D+N
		query = torch.cat([query_lb, query], dim=2)  # B, NQ*QL, D+N
		D = D+N

		# cat 1 query and K support together
		seq = torch.cat([query, support], dim=1)  # (B, NQ*QL+N*K*SL, D)
		seq_mask = torch.cat([query_mask, support_mask], dim=1)  # (B, NQ*QL+N*K*SL)
		seq = self.transformer(src=seq.transpose(0, 1),
									src_key_padding_mask=~seq_mask.bool()).transpose(0, 1)
		query_ = seq[:, :NQ*QL]  # B, NQ*QL, D
		support_ = seq[:, NQ*QL:]  # B, N*K*SL, D
		query_mask = seq_mask[:, :NQ*QL].contiguous().view(B*NQ, QL)  # B*NQ, QL
		support_mask = seq_mask[:, NQ*QL:].contiguous().view(B*N*K, SL)  # B*N*K, SL

		query = self.fuse(query, query_, dim=2).contiguous().view(B*NQ, QL, 4*D)   # 4D
		support = self.fuse(support, support_, dim=2).contiguous().view(B*N*K, SL, 4*D)  # 4D

		# pool sentence into 1 vector
		query = self.seq_pooling(query, query_mask).contiguous().view(B, NQ, 8*D)  # 8D
		support = self.seq_pooling(support, support_mask).contiguous().view(B, N*K, 8*D)  # 8D

		# dim reduce
		D = self.hidden_size
		query = self.fc(query)  # D
		support = self.fc(support)  # D

		support_lb = support_lb_emb[None, :, None, :].expand(B, -1, K, -1).contiguous().view(B, N*K, -1)  # B, N*K, N
		query_lb = query_lb_emb[None, None, :].expand(B, NQ, -1).view(B, NQ, -1)
		support = torch.cat([support_lb, support], dim=2)  # B, N*K, D+N
		query = torch.cat([query_lb, query], dim=2)  # B, NQ, D+N
		D = D + N

		w = torch.cat([support, query], dim=1)  # B, N*K+NQ, D+N
		w = self.gnn(src=w.transpose(0, 1)).transpose(0, 1)  # B, N*K+NQ, D+N
		w_q = w[:, N * K:]
		w = w[:, :N * K]

		w = w.contiguous().view(B, N, K, -1).mean(2).unsqueeze(1).expand(-1, NQ, N, -1)  # B, total_Q, N, D
		w = w.transpose(-1, -2)
		logits = torch.matmul(query.contiguous().view(-1, 1, D),
							  w.contiguous().view(-1, D, N)).squeeze().contiguous()
		_, pred = torch.max(logits.view(-1, N), 1)

		return logits, pred

