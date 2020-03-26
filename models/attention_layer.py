import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_bert import BertModel


class SelfAttention(nn.Module):
	def __init__(self, hidden_size, num_heads, dropout=0.1):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_attention_heads = num_heads
		self.attention_head_size = int(hidden_size / num_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size
		self.query = nn.Linear(hidden_size, self.all_head_size)
		self.key = nn.Linear(hidden_size, self.all_head_size)
		self.value = nn.Linear(hidden_size, self.all_head_size)
		self.dropout = nn.Dropout(dropout)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self,
				hidden_states,
				attention_mask=None,
				attend_hidden_states=None,
				attend_attention_mask=None):

		mixed_query_layer = self.query(hidden_states)
		if attend_hidden_states is not None:
			mixed_key_layer = self.key(attend_hidden_states)
			mixed_value_layer = self.value(attend_hidden_states)
			# attend_attention_mask = attend_attention_mask[:, None, None, :]
			# attend_attention_mask = (1.0 - attend_attention_mask.float()) * -10000
			attention_mask = attend_attention_mask
		else:
			mixed_key_layer = self.key(hidden_states)
			mixed_value_layer = self.value(hidden_states)
			# if attention_mask is not None:
			# 	attention_mask = attention_mask[:, None, None, :]
			# 	attention_mask = (1.0 - attention_mask.float()) * -10000

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
		print(attention_probs[0][0])
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


class AttentionLayer(nn.Module):
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
		# attention_output = self_outputs[0]
		outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
		return outputs
