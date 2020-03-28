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
		[pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
		if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
	position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # apply sin on 0th,2nd,4th...emb_dim
	position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # apply cos on 1st,3rd,5th...emb_dim
	return torch.from_numpy(position_enc).float()

class SENTSEL(fewshot_re_kit.framework.FewShotREModel):

	def __init__(self, sentence_encoder, hidden_size=230, dropout=0.2):
		fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
		self.hidden_size = hidden_size
		self.drop = nn.Dropout(dropout)
		pos_embs = position_encoding_init(sentence_encoder.max_length, hidden_size)
		self.positional_embeddings = nn.Embedding.from_pretrained(pos_embs, freeze=False)

		# for sentence level self-attention
		layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=5, dim_feedforward=384, dropout=0.1)
		self.transformer = nn.TransformerEncoder(encoder_layer=layer, num_layers=4)
		# gnn
		# self.gnn = nn.TransformerEncoder(encoder_layer=layer, num_layers=2)


	def forward(self, support, query, N, K, Q):
		'''
		support: Inputs of the support set.
		query: Inputs of the query set.
		N: Num of classes
		K: Num of instances for each class in the support set
		Q: Num of instances in the query set
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

		# add positional embeddings for transformer
		support_pos_ids = torch.arange(SL, device=support.device).view(1, -1).expand(support.shape[0], -1)  # B*N*K, SL
		query_pos_ids = torch.arange(QL, device=query.device).view(1, -1).expand(query.shape[0], -1)  # B*N*K, QL
		support_pos_emb = self.positional_embeddings(support_pos_ids)
		query_pos_emb = self.positional_embeddings(query_pos_ids)
		support = support + support_pos_emb
		query = query + query_pos_emb

		# get batch size and NQ
		support = support.contiguous().view(-1, N*K, SL, D)  # (B, N * K, SL, D)
		B = support.shape[0]  # batch size
		query = query.contiguous().view(B, -1, QL, D)  # (B, NQ, QL, D)
		NQ = query.shape[1]  # num of instances for each batch in query set

		########### sentence level cross attention ##########
		# first expand query N times for every class
		query = query.contiguous().view(B, NQ, 1, QL, D).expand(-1, -1, N, -1, -1).contiguous().view(-1, QL, D)  # (B*NQ*N, QL, D)
		query_mask = query_mask.contiguous().view(B, NQ, 1, QL).expand(-1, -1, N, -1).contiguous().view(-1, QL)  # (B*NQ*N, QL)
		# cat K support instance into one line, and repeat NQ times
		support = support.contiguous().view(B, 1, N*K, SL, D).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, K*SL, D)  # (B*NQ*N, K*SL, D)
		support_mask = support_mask.contiguous().view(B, 1, N, K*SL).expand(-1, NQ, -1, -1).contiguous().view(-1, K*SL)  #(B*NQ*N, K*SL)

		# cat 1 query and K support together
		sentences = torch.cat([query, support], dim=1)  # (B*NQ*N, QL+K*SL, D)
		sentences_mask = torch.cat([query_mask, support_mask], dim=1)  # (B*NQ*N, QL+K*SL)
		sentences = self.cross_attn(src=sentences.transpose(0, 1), src_key_padding_mask=~sentences_mask.bool()).transpose(0, 1)
		query = sentences[:, :QL]  # B*NQ*N, QL, D
		support = sentences[:, QL:]  # B*NQ*N, K*SL, D
		del sentences
		query_mask = sentences_mask[:, :QL]   # B*NQ*N, QL
		support_mask = sentences_mask[:, QL:]  # B*NQ*N, K*SL
		del sentences_mask


		# pool sentence into 1 vector
		support = support - (1.0 - support_mask.float()[:, :, None]) * 10000
		query = query - (1.0 - query_mask.float()[:, :, None]) * 10000
		support = support.view(B * NQ * N * K, SL, D).max(1)[0]  # (B*NQ*N*K, D)
		query = query.max(1)[0]  # (B*NQ*N, D)

		# fea_att_score = support.view(B * N, 1, K, self.hidden_size)  # (B * N, 1, K, D)
		# fea_att_score = F.relu(self.conv1(fea_att_score))  # (B * N, 32, K, D)
		# fea_att_score = F.relu(self.conv2(fea_att_score))  # (B * N, 64, K, D)
		# fea_att_score = self.drop(fea_att_score)
		# fea_att_score = self.conv_final(fea_att_score)  # (B * N, 1, 1, D)
		# fea_att_score = F.relu(fea_att_score)
		# fea_att_score = fea_att_score.view(B, N, self.hidden_size).unsqueeze(1)  # (B, 1, N, D)

		support_idxs = torch.arange(0, N, device=support.device)[None, :, None].expand(B, -1, K).contiguous().view(B, N * K)
		query_idxs = torch.tensor([N], device=query.device)[None, :, None].expand(B, -1, NQ).contiguous().view(B,
																													total_Q)

		support = support.contiguous().view(B, NQ, N, K, D)  # (B, NQ, N, K, D)
		query = query.contiguous().view(B, NQ, N, D)  # B, NQ, N, D
		# prototype
		if K != 1:
			# instance-level attention
			support_for_att = self.fc(support)
			query_for_att = self.fc(query).unsqueeze(3).expand(-1, -1, -1, K, D)  # B, NQ, N, K, D
			ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1)  # (B, NQ, N, K)
			support_proto = (support * ins_att_score.unsqueeze(4).expand(-1, -1, -1, -1, self.hidden_size)).sum(3)  # (B, NQ, N, D)
			# support_proto = support.mean(3)
		else:
			# no instance attention
			support_proto = support.squeeze(3)
		# Prototypical Networks
		logits = -self.__batch_dist__(support_proto, query, None).view(-1, N)



		_, pred = torch.max(logits.view(-1, N), 1)
		# print(logits)
		return logits, pred


