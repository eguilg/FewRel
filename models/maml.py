import sys

sys.path.append('..')
import fewshot_re_kit
from fewshot_re_kit.network.embedding import Embedding
from fewshot_re_kit.network.encoder import Encoder
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


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
	return total_norm / counter


class MAML(fewshot_re_kit.framework.FewShotREModel):

	def __init__(self, embedding, N, max_length, hidden_size=230):
		'''
		N: num of classes
		K: num of instances for each class
		word_vec_mat, max_length, hidden_size: same as sentence_encoder
		'''
		fewshot_re_kit.framework.FewShotREModel.__init__(self, None)
		self.max_length = max_length
		self.hidden_size = hidden_size
		self.N = N

		self.meta_update_step = 5
		self.meta_lr = .5

		self.embedding = embedding

		self.encoder = Encoder(max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=hidden_size)
		self.fc = nn.Linear(hidden_size, N + 1, bias=False)

		self.fast_conv_W = None
		self.fast_fc_W = None


	def predict(self, x, size, use_fast=False):
		# x = self.embedding(inputs)
		if use_fast:
			x = F.relu(F.conv1d(x.transpose(-1, -2), self.fast_conv_W, padding=1)).max(-1)[0]
			output = F.linear(x, self.fast_fc_W)
		else:
			x = self.encoder(x)
			output = self.fc(x)
		return output.view(size)

	def forward(self, support, query, N, K, NQ):
		'''
		support: Inputs of the support set.
		query: Inputs of the query set.
		N: Num of classes
		K: Num of instances for each class in the support set
		Q: Num of instances for each class in the query set
		'''
		support = self.embedding(support)  # B, Len, D
		support = support.view(-1, N*K, self.max_length, support.shape[-1])
		B = support.size(0)
		query = self.embedding(query)
		query = query.view(B, NQ, -1, support.shape[-1])
		# learn fast parameters for attention encoder


		# tmp_label = Variable(torch.tensor([[x] * K for x in range(N)] * B, dtype=torch.long).cuda())
		tmp_label = torch.arange(0, N, device=support.device)[None, :, None].expand(B, -1, K).contiguous().view(B, N * K)

		logits_meta = []
		if not self.training:
			update_step = self.meta_update_step * 2
		else:
			update_step = self.meta_update_step
		for i in range(B):
			losses = []
			# 1. run the i-th task and compute loss for k=0
			logits = self.predict(support[i], (N * K, N + 1))
			loss = F.cross_entropy(logits, tmp_label[i])
			fc_grad = torch.autograd.grad(loss, self.fc.parameters(), retain_graph=True)
			conv_grad = torch.autograd.grad(loss, self.encoder.conv.parameters())
			clip_grad_by_norm_(fc_grad, 5.0)
			clip_grad_by_norm_(conv_grad, 5.0)
			self.fast_fc_W = list(self.fc.parameters())[0] - self.meta_lr * fc_grad[0]
			self.fast_conv_W = list(self.encoder.parameters())[0] - self.meta_lr * conv_grad[0]

			losses.append(loss)
			for k in range(1, update_step):
				# 1. run the i-th task and compute loss for k=1~K-1
				logits = self.predict(support[i], (N*K, N+1), True)
				loss = F.cross_entropy(logits, tmp_label[i])
				# 2. compute grad on theta_pi
				fc_grad = torch.autograd.grad(loss, [self.fast_fc_W], retain_graph=True)
				conv_grad = torch.autograd.grad(loss, [self.fast_conv_W])
				clip_grad_by_norm_(fc_grad, 5.0)
				clip_grad_by_norm_(conv_grad, 5.0)
				# print(grad.max())
				# 3. theta_pi = theta_pi - train_lr * grad
				self.fast_fc_W = self.fast_fc_W - self.meta_lr * fc_grad[0]
				self.fast_conv_W = self.fast_conv_W - self.meta_lr * conv_grad[0]
				losses.append(loss.item())
			# 4. logits on query set
			logits_meta.append(self.predict(query[i], (-1, N+1), True))
		# print(np.argmin(losses))
		logits = torch.stack(logits_meta, dim=0)  # B*total_Q, N+1
		# logits[:, :, -1] = 0.5
		_, pred = torch.max(logits.view(-1, N+1), 1)
		return logits, pred
