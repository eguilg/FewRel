import sys

sys.path.append('..')
import math

import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers.modeling_bert import BertModel
import fewshot_re_kit


class SelfAttention(nn.Module):
	def __init__(self, hidden_size, num_heads, dropout=0.1):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_attention_heads = num_heads
		self.attention_head_size = int(hidden_size / num_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size
		self.query = nn.Linear(self.hidden_size, self.all_head_size)
		self.key = nn.Linear(hidden_size, self.all_head_size)
		self.value = nn.Linear(hidden_size, self.all_head_size)
		self.dropout = nn.Dropout(dropout)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, hidden_states, attention_mask=None,
				attend_hidden_states=None,
				attend_attention_mask=None):

		if attend_hidden_states is not None:
			mixed_key_layer = self.key(attend_hidden_states)
			mixed_value_layer = self.value(attend_hidden_states)
			attention_mask = attend_attention_mask
		else:
			mixed_key_layer = self.key(hidden_states)
			mixed_value_layer = self.value(hidden_states)
		mixed_query_layer = self.query(hidden_states)
		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)

		if attention_mask is not None:
			attention_scores = attention_scores + attention_mask
		# Normalize the attention scores to probabilities.
		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)

		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)

		outputs = (context_layer, attention_probs)

		return outputs


class SelfOutput(nn.Module):
	def __init__(self, in_dim, out_dim, dropout=0.1):
		super().__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.dense = nn.Linear(in_dim, in_dim)
		self.LayerNorm = nn.LayerNorm(in_dim, eps=1e-12)
		self.dropout = nn.Dropout(dropout)
		if out_dim != in_dim:
			self.dense2 = nn.Linear(in_dim, out_dim)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		if self.out_dim != self.in_dim:
			hidden_states = self.dense2(F.leaky_relu(hidden_states))
		return hidden_states


class AtentionLayer(nn.Module):
	def __init__(self, in_dim, out_dim, num_heads, dropout=0.1):
		super().__init__()
		self.self = SelfAttention(in_dim, num_heads, dropout)
		self.output = SelfOutput(in_dim, out_dim, dropout)

	def forward(self, hidden_states,
				attention_mask=None,
				attend_hidden_states=None,
				attend_attention_mask=None):
		if attention_mask is not None:
			attention_mask = attention_mask[:, None, None, :]
			attention_mask = (1.0 - attention_mask.float()) * -10000
		if attend_attention_mask is not None:
			attend_attention_mask = attend_attention_mask[:, None, None, :]
			attend_attention_mask = (1.0 - attend_attention_mask.float()) * -10000
		self_outputs = self.self(hidden_states, attention_mask, attend_hidden_states, attend_attention_mask)
		attention_output = self.output(self_outputs[0], hidden_states)
		outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
		return outputs


class GOG(fewshot_re_kit.framework.FewShotREModel):

	def __init__(self, sentence_encoder, N, hidden_size=230, num_heads=5, num_layers=2, na_rate=0):
		'''
		N: Num of classes
		'''
		fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
		self.hidden_size = hidden_size
		self.na_rate = na_rate
		self.num_layers = num_layers
		extra = 1
		while (hidden_size + N + extra) % num_heads != 0:
			extra += 1
		emb_w = torch.eye(N+extra)
		emb_w[N, :] = 0.5
		self.label_embeddings = nn.Embedding.from_pretrained(emb_w, freeze=False)

		in_dim = hidden_size + N + extra
		self.selfatt_layers = nn.ModuleList([
			AtentionLayer(hidden_size, hidden_size, num_heads, dropout=0.1) for _ in range(num_layers)
		])
		self.crossatt_layers = nn.ModuleList([
			AtentionLayer(in_dim, hidden_size, num_heads, dropout=0.1) for _ in range(num_layers)
		])
		# self.output = AtentionLayer(in_dim, N+1, num_heads, dropout=0.1)

	def forward(self, support, query, N, K, total_Q):
		'''
		support: Inputs of the support set.
		query: Inputs of the query set.
		N: Num of classes
		K: Num of instances for each class in the support set
		total_Q: Num of instances in the query set
		'''
		s_mask = support['mask']
		q_mask = query['mask']
		L = s_mask.shape[1]
		B = s_mask.shape[0] // (N*K)
		D = self.hidden_size
		M = N*K + total_Q
		s_mask = s_mask.view(B, N*K, L)
		q_mask = q_mask.view(B, total_Q, L)

		support_seq = self.sentence_encoder(support, pool=False).view(B, N * K, L, D)
		query_seq = self.sentence_encoder(query, pool=False).view(B, total_Q, L, D)
		hidden_seq = torch.cat([support_seq, query_seq], dim=1)  # B, M, L, D
		mask = torch.cat([s_mask, q_mask], dim=1)  # B, M, L
		support_idxs = torch.arange(0, N, device=s_mask.device)[None, :, None].expand(B, -1, K).contiguous().view(B,
																												  N * K)
		query_idxs = torch.tensor([N], device=q_mask.device)[None, :, None].expand(B, -1, total_Q).contiguous().view(B,
																											 total_Q)
		label_emb = torch.cat([self.label_embeddings(support_idxs), self.label_embeddings(query_idxs)], dim=1)  # B, M, Dlb

		hidden_vec = hidden_seq - (1.0 - mask.float()[:, :, :, None]) * 10000
		hidden_vec = hidden_vec.max(2)[0]  # B, M, D
		hidden_seq = hidden_seq.contiguous().view(B * M, L, D)  # B*M, L, D

		mask = mask.contiguous().view(B * M, L)  # B*M, L

		for i in range(self.num_layers):
			new_seq = self.selfatt_layers[i](hidden_seq)[0]  # B*M, L, D
			hidden_seq = new_seq# + hidden_seq
			new_vec = hidden_seq - (1.0 - mask.float()[:, :, None]) * 10000  # B*M, D
			new_vec = new_vec.max(1)[0].contiguous().view(B, M, D)  # B, M, D
			hidden_vec = new_vec# + hidden_vec
			hidden_vec = torch.cat([label_emb, hidden_vec], dim=-1)  # B, M, D+Dlb
			hidden_vec = self.crossatt_layers[i](hidden_vec)[0]


		w_q = hidden_vec[:, N * K:]  # B, total_Q, D
		w = hidden_vec[:, :N * K]  # B, N*K, D

		w = w.contiguous().view(B, N, K, D).mean(2).unsqueeze(1).expand(-1, total_Q, N, -1)  # B, total_Q, N, D
		query_vec = query_seq - (1.0 - q_mask.float()[:, :, :, None]) * 10000
		query_vec = query_vec.max(2)[0].view(B*total_Q, D)
		if self.na_rate > 0:
			w = torch.cat([w, w_q.unsqueeze(2)], dim=2).transpose(-1, -2)  # B, total_Q, D, N+1
			logits = torch.matmul(query_vec.contiguous().view(-1, 1, D),
								  w.contiguous().view(-1, D, N + 1)).squeeze().contiguous()
		else:
			w = w.transpose(-1, -2)
			logits = torch.matmul(query_vec.contiguous().view(-1, 1, D),
								  w.contiguous().view(-1, D, N)).squeeze().contiguous()


		_, pred = torch.max(logits, 1)
		return logits, pred
