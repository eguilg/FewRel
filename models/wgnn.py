import sys

sys.path.append('..')
import math

import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

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

	def forward(self, hidden_states):
		mixed_query_layer = self.query(hidden_states)
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)
		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)

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

	def forward(self, hidden_states):
		self_outputs = self.self(hidden_states)
		attention_output = self.output(self_outputs[0], hidden_states)
		outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
		return outputs


class WGNN_core(nn.Module):
	def __init__(self, in_dim, out_dim, num_heads, num_layer=2, dropout=0.1):
		super().__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.layers = nn.ModuleList([
			AtentionLayer(in_dim, in_dim, num_heads, dropout) for _ in range(num_layer)
		])
		self.last = AtentionLayer(in_dim, out_dim, num_heads, dropout)

	def forward(self, hidden_states):
		for i, layer_module in enumerate(self.layers):
			out = layer_module(hidden_states)[0]
			hidden_states = hidden_states + out
		hidden_states = self.last(hidden_states)[0]
		return hidden_states


class WGNN(fewshot_re_kit.framework.FewShotREModel):

	def __init__(self, sentence_encoder, N, hidden_size=230, num_heads=4, na_rate=0):
		'''
		N: Num of classes
		'''
		fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
		self.hidden_size = hidden_size
		self.na_rate = na_rate
		extra = 1
		while (hidden_size + N + extra) % num_heads != 0:
			extra += 1
		emb_w = torch.eye(N+extra)
		emb_w[N, :] = 0.5

		self.label_embeddings = nn.Embedding.from_pretrained(emb_w, freeze=False)
		self.gnn_obj = WGNN_core(hidden_size + N + extra, hidden_size, num_heads, num_layer=3)


	def forward(self, support, query, N, K, total_Q):
		'''
		support: Inputs of the support set.
		query: Inputs of the query set.
		N: Num of classes
		K: Num of instances for each class in the support set
		total_Q: Num of instances in the query set
		'''
		support = self.sentence_encoder(support)
		query = self.sentence_encoder(query)
		support = support.view(-1, N*K, self.hidden_size)
		query = query.view(-1, total_Q, self.hidden_size)

		B = support.size(0)
		# NQ = query.size(1)
		D = self.hidden_size

		support_idxs = torch.arange(0, N, device=support.device)[None, :, None].expand(B, -1, K).contiguous().view(B, N*K)
		query_idxs = torch.tensor([N], device=query.device)[None, :, None].expand(B, -1, total_Q).contiguous().view(B, total_Q)

		support_W_init = self.label_embeddings(support_idxs)  # B, N*K, D
		query_W_init = self.label_embeddings(query_idxs)  # B, total_Q, D

		support_W_init = torch.cat([support_W_init, support], dim=-1)  # B, N*K, 2D
		query_W_init = torch.cat([query_W_init, query], dim=-1)  # B, total_Q, 2D

		w = torch.cat([support_W_init, query_W_init], dim=1)  # B, N*K + total_Q, 2D
		w = self.gnn_obj(w)  # (B, N*K+total_Q, D)
		# print(w.shape, N*K, total_Q)
		w_q = w[:, N*K:]
		w = w[:, :N*K]


		# w = w.contiguous().view(B, N, K, -1).mean(2).unsqueeze(1).expand(-1, total_Q, N, -1).transpose(-1, -2)  # B, total_Q, N, D
		# logits = torch.matmul(query.view(-1, 1, D), w.contiguous().view(-1, D, N)).squeeze()

		w = w.contiguous().view(B, N, K, -1).mean(2).unsqueeze(1).expand(-1, total_Q, N, -1)  # B, total_Q, N, D
		if self.na_rate > 0:
			w = torch.cat([w, w_q.unsqueeze(2)], dim=2).transpose(-1, -2)  # B, total_Q, D, N+1
			logits = torch.matmul(query.contiguous().view(-1, 1, D), w.contiguous().view(-1, D, N+1)).squeeze().contiguous()
		else:
			w = w.transpose(-1, -2)
			logits = torch.matmul(query.contiguous().view(-1, 1, D), w.contiguous().view(-1, D, N)).squeeze().contiguous()
		# print(logits.shape)
		_, pred = torch.max(logits, 1)
		return logits, pred
