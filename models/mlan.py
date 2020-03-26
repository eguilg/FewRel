import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from .attention_layer import SelfAttention


class MLAN(fewshot_re_kit.framework.FewShotREModel):

	def __init__(self, sentence_encoder, shots, hidden_size=230):
		fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
		self.hidden_size = hidden_size
		self.drop = nn.Dropout()
		# for sentence level self-attention
		# self.self_attn = nn.TransformerEncoderLayer(hidden_size=hidden_size,  num_heads=5, dropout=0.1)
		# for sentence level cross attention
		self.cross_attn = nn.TransformerDecoderLayer(d_model=hidden_size,  nhead=5, dim_feedforward=512, dropout=0.1)
		# for sentence level self-attention
		# self.agg_attn = nn.TransformerEncoderLayer(d_model=hidden_size,  nhead=5, dim_feedforward=512, dropout=0.1)

		# for instance-level attention
		self.fc = nn.Linear(hidden_size, hidden_size, bias=True)
		# self.inst_attn = nn.TransformerEncoderLayer(d_model=hidden_size,  nhead=5, dim_feedforward=512, dropout=0.1)
		# for feature-level attention
		# self.conv1 = nn.Conv2d(1, 32, (shots, 1), padding=(shots // 2, 0))
		# self.conv2 = nn.Conv2d(32, 64, (shots, 1), padding=(shots // 2, 0))
		# self.conv_final = nn.Conv2d(64, 1, (shots, 1), stride=(shots, 1))

	def __dist__(self, x, y, dim, score=None):
		if score is None:
			return (torch.pow(x - y, 2)).sum(dim)
		else:
			return (torch.pow(x - y, 2) * score).sum(dim)

	def __batch_dist__(self, S, Q, score=None):
		# print(S.shape, Q.shape, score.shape)
		return self.__dist__(S, Q, 3, score)

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
		SL = support_mask.sum(1).max().item()  # support batch max length
		QL = query_mask.sum(1).max().item()  # query batch max length
		# print(support['mask'].sum(1).max())
		D = self.hidden_size  # hidden size
		support = self.sentence_encoder(support, pool=False)  # (B * N * K, L, D)
		query = self.sentence_encoder(query, pool=False)  # （B * NQ, L, D）
		support = support[:, :SL]  # cut pads
		support_mask = support_mask[:, :SL]
		query = query[:, :QL]  # cut pads
		query_mask = query_mask[:, :QL]
		support = support.contiguous().view(-1, N*K, SL, D)  # (B, N * K, SL, D)
		B = support.shape[0]  # batch size
		query = query.contiguous().view(B, -1, QL, D)  # (B, NQ, QL, D)
		NQ = query.shape[1]  # num of instances for each batch in query set

		# sentence level self attention
		support = support.view(B*N, K*SL, D)
		support_mask = support_mask.contiguous().view(B*N, K*SL)
		query = query.view(B*NQ, QL, D)
		# support = self.self_attn(hidden_states=support, attention_mask=support_mask)[0]  # (B*N, K*SL, D)
		# query = self.self_attn(hidden_states=query, attention_mask=query_mask)[0]  # (B*NQ, QL, D)

		# sentence level cross attention
		support_for_cross = support.contiguous().view(B, 1, N*K, SL, D).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, K*SL, D)  # (B*NQ*N, K*SL, D)
		query_for_cross = query.contiguous().view(B, NQ, 1, QL, D).expand(-1, -1, N, -1, -1).contiguous().view(-1, QL, D)  # (B*NQ*N, QL, D)
		support_mask_for_cross = support_mask.contiguous().view(B, 1, N, K*SL).expand(-1, NQ, -1, -1).contiguous().view(-1, K*SL)  #(B*NQ*N, K*SL)
		query_mask_for_cross = query_mask.contiguous().view(B, NQ, 1, QL).expand(-1, -1, N, -1).contiguous().view(-1, QL)  # (B*NQ*N, QL)

		support = self.cross_attn(
			tgt=support_for_cross.transpose(0, 1),
			memory=query_for_cross.transpose(0, 1),
			tgt_key_padding_mask=~support_mask_for_cross.bool(),
			memory_key_padding_mask=~query_mask_for_cross.bool()
		).transpose(0, 1)  # (B*NQ*N, K*SL, D)
		query = self.cross_attn(
			tgt=query_for_cross.transpose(0, 1),
			memory=support_for_cross.transpose(0, 1),
			tgt_key_padding_mask=~query_mask_for_cross.bool(),
			memory_key_padding_mask=~support_mask_for_cross.bool()
		).transpose(0, 1)  # (B*NQ*N, QL, D)

		# support = support_for_cross
		# query = query_for_cross
		del support_for_cross, query_for_cross

		# pool sentence into 1 vector
		support = support - (1.0 - support_mask_for_cross.float()[:, :, None]) * 10000
		query = query - (1.0 - query_mask_for_cross.float()[:, :, None]) * 10000
		support = support.view(B * NQ * N * K, SL, D).max(1)[0]  # (B*NQ*N*K, D)
		query = query.max(1)[0]  # (B*NQ*N, D)
		del support_mask_for_cross, query_mask_for_cross


		# instance-level attention
		# support = support.view(B*NQ, N*K, D)
		# query = query.view(B*NQ, )

		support = support.contiguous().view(B, NQ, N, K, D)  # (B, NQ, N, K, D)
		support_for_att = self.fc(support)
		query = query.contiguous().view(B, NQ, N, D)  # B, NQ, N, D
		query_for_att = self.fc(query).unsqueeze(3).expand(-1, -1, -1, K, D)  # B, NQ, N, K, D
		ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1)  # (B, NQ, N, K)
		support_proto = (support * ins_att_score.unsqueeze(4).expand(-1, -1, -1, -1, self.hidden_size)).sum(3)  # (B, NQ, N, D)

		# Prototypical Networks
		# print(((support_proto * query).sum(-1)<0).sum())
		# logits = -torch.tanh(support_proto *  query).sum(-1).view(-1, N)
		logits = -self.__batch_dist__(support_proto, query, None).view(-1, N)
		_, pred = torch.max(logits.view(-1, N), 1)
		# print(logits)
		return logits, pred


