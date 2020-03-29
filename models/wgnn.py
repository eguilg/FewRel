import sys

sys.path.append('..')
import math
import numpy as np
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

import fewshot_re_kit

def clip_grad_by_norm_(grad, max_norm):

	"""
	in-place gradient clipping.
	:param grad: list of gradients
	:param max_norm: maximum norm allowable
	:return:
	"""
	total_norm = 0
	counter = 0
	for g in grad:
		param_norm = g.data.norm(2)
		total_norm += param_norm.item() ** 2
		counter += 1
	total_norm = total_norm ** (1. / 2)
	clip_coef = max_norm / (total_norm + 1e-6)
	if clip_coef < 1:
		for g in grad:
			g.data.mul_(clip_coef)
	return total_norm/counter


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
		self.gnn_obj = WGNN_core(hidden_size + N + extra, hidden_size, num_heads, num_layer=2)


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
		w_q = w[:, N*K:]
		w = w[:, :N*K]

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



class MetaWGNN(fewshot_re_kit.framework.FewShotREModel):

	def __init__(self, sentence_encoder, N, hidden_size=230, num_heads=4, na_rate=0,
				 meta_lr=1.0, meta_update_step=5):
		'''
		N: Num of classes
		'''
		fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
		self.hidden_size = hidden_size
		self.na_rate = na_rate
		extra = 1
		while (hidden_size*2 + N + extra) % num_heads != 0:
			extra += 1
		emb_w = torch.eye(N+extra)
		emb_w[N, :] = 0.5
		self.fc = nn.Linear(hidden_size, N+1, bias=False)
		self.label_embeddings = nn.Embedding.from_pretrained(emb_w, freeze=False)
		self.gnn_obj = WGNN_core(hidden_size*2 + N + extra, hidden_size, num_heads, num_layer=2)

		self.meta_lr = meta_lr
		self.meta_update_step = meta_update_step
	def loss(self, logits, label):
		'''
		logits: Logits with the size (..., class_num)
		label: Label with whatever size.
		return: [Loss] (A single value)
		'''
		logits, J_incon = logits
		N = logits.size(-1)
		return self.cost(logits.view(-1, N), label.view(-1)) + J_incon

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
		logits_meta = []
		fast_weights = []
		if not self.training:
			update_step = self.meta_update_step * 2
		else:
			update_step = self.meta_update_step
		for i in range(B):
			losses = []
			# 1. run the i-th task and compute loss for k=0
			logits = self.fc(support[i])  # N*K, N+1
			loss = F.cross_entropy(logits, support_idxs[i])
			grad = torch.autograd.grad(loss, self.fc.parameters())
			clip_grad_by_norm_(grad, 5.0)
			grad = grad[0]  # N+1, D  (grade for linear weight)
			fast_weight = list(self.fc.parameters())[0] - self.meta_lr * grad
			losses.append(loss)
			for k in range(1, update_step):
				# 1. run the i-th task and compute loss for k=1~K-1
				logits = F.linear(support[i], fast_weight)
				loss = F.cross_entropy(logits, support_idxs[i])
				# 2. compute grad on theta_pi
				grad = torch.autograd.grad(loss, [fast_weight])
				clip_grad_by_norm_(grad, 5.0)
				grad = grad[0]
				# print(grad.max())
				# 3. theta_pi = theta_pi - train_lr * grad
				fast_weight = fast_weight - self.meta_lr * grad
				losses.append(loss.item())
			# 4. logits on query set
			logits_meta.append(F.linear(query[i], fast_weight))
			fast_weights.append(fast_weight)
			# print(np.argmin(losses))
		logits_meta = torch.stack(logits_meta, dim=0).view(B*total_Q, -1)  # B*total_Q, N+1
		fast_weights = torch.stack(fast_weights, dim=0)  # B, N+1, D

		# if not self.is_training:
		# print(logits_meta.shape)
		# print(fast_weight)
		# if self.training:
		# 	fast_weights = fast_weights + torch.randn_like(fast_weights, device=fast_weights.device) / math.sqrt(self.meta_lr)
		support_w = fast_weights[:, :N].unsqueeze(2).expand(-1, -1, K, -1).contiguous().view(B, N*K, D)
		query_w = fast_weights[:, -1:].expand(-1, total_Q, -1)
		support_lb_emb = self.label_embeddings(support_idxs)  # B, N*K, D
		query_lb_emb = self.label_embeddings(query_idxs)  # B, total_Q, D

		support_W_init = torch.cat([support_lb_emb, support_w, support], dim=-1)    # B, N*K, D_
		# support_W_init = support_W_init.unsqueeze(1).expand(-1, total_Q, -1, -1).contiguous().view(B*total_Q, N*K, -1)  # B, total_Q, N*K, D_
		query_W_init = torch.cat([query_lb_emb, query_w, query], dim=-1)  # B, total_Q, D_
		# query_W_init = query_W_init.contiguous().view(B*total_Q, 1, -1)  # B*total_Q, 1, D_

		w = torch.cat([support_W_init, query_W_init], dim=1)  # B, N*K + total_Q, Dlb
		# w = torch.tensor(w, requires_grad=False)
		w = self.gnn_obj(w)  # (B, N*K+total_Q, D)
		w_q = w[:, N * K:]
		w = w[:, :N * K]

		w = w.contiguous().view(-1, K, D)  # B*N, K, D
		proto = w.mean(1)  # prototype  B*N, D

		J_incon = torch.sum((proto.unsqueeze(1) * w), 2)
		J_incon = 1 - (J_incon / (torch.norm(proto.unsqueeze(1), dim=-1) * torch.norm(w, dim=-1)))
		J_incon = J_incon.mean()
		proto = proto.view(-1, N, D).unsqueeze(1).expand(-1, total_Q, N, D)  # B, total_Q, N, D
		w_q = w_q.contiguous().view(-1, total_Q, 1, D)
		logits = torch.sum((w_q * proto), dim=3).view(-1, N)
		# print(logits.shape)
		logits_meta[:, :logits.shape[1]] = (logits_meta[:, :logits.shape[1]] + logits) / 2
		# logits = (logits + logits_meta) / 2
		logits = logits_meta

		if self.na_rate == 0:
			logits = logits[:, :N]

		_, pred = torch.max(logits, 1)
		return (logits, J_incon), pred