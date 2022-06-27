#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMRNNDecoderAttention(nn.Module):
	"""docstring for LSTMRNNDecoderAttention"""
	def __init__(self, en_hidden_size, de_hidden_size, device):
		super(LSTMRNNDecoderAttention, self).__init__()
		self.device = device
		self.attn = torch.nn.Linear(
			de_hidden_size*2 + en_hidden_size, 
			de_hidden_size)
		self.v = torch.nn.Parameter(torch.rand(de_hidden_size))
		self.v.data.normal_(mean=0, std=1./self.v.size(0)**(1./2.))

	def score(self, hidden, encoder_output):
		# hidden: batch_size, max_seq_len, de_hidden_size*2
		# encoder_output: batch_size, src_seq_len, hidden_size
		# batch_size, max_seq_len, de_hidden_size
		energy = self.attn(torch.cat([hidden, encoder_output], 2))
		# batch_size, de_hidden_size, src_seq_len
		energy = torch.tanh(energy).transpose(2, 1)
		# batch_size, 1, de_hidden_size
		v = self.v.repeat(encoder_output.shape[0], 1).unsqueeze(1)
		# batch_size, 1, max_seq_len
		energy = torch.bmm(v, energy)
		# batch_size, max_seq_len
		return energy.squeeze(1)

	def forward(self, hidden, encoder_output, x_lens):
		# hidden (h, c): 1, batch_size, num_layers*hidden_size
		# encoder_output: batch_size, src_seq_len, hidden_size
		# x_lens: batch_size
		# batch_size, de_hidden_size*2
		hidden = torch.cat(hidden, 2).squeeze(0)
		# batch_size, max_seq_len, de_hidden_size*2
		hidden = hidden.repeat(encoder_output.shape[1], 1, 1).transpose(0, 1)
		# batch_size, max_src_seq_len
		attn_energies = self.score(hidden, encoder_output)
		# max_seq_len
		idx = torch.arange(
			end=encoder_output.shape[1]
			, dtype=torch.float
			, device=self.device)
		# batch_size, max_seq_len
		idx = idx.unsqueeze(0).expand(attn_energies.shape)
		# batch size, max_seq_len
		x_lens = x_lens.unsqueeze(-1).expand(attn_energies.shape).to(self.device)
		mask = idx < x_lens
		attn_energies[~mask] = float('-inf') 
		# batch_size, 1, max_src_seq_len
		return F.softmax(attn_energies, dim=1).unsqueeze(1)


class SelfDecoderAttention(nn.Module):
	"""docstring for SelfDecoderAttention"""
	def __init__(self, en_hidden_size, de_embedding_size, de_hidden_size, max_seq_len, device):
		super(SelfDecoderAttention, self).__init__()
		self.device = device
		self.max_seq_len = max_seq_len
		self.attn = nn.Linear(
			in_features=en_hidden_size*2 + de_embedding_size
			, out_features=max_seq_len)

	def forward(self, xs, x_lens, hidden):
		# xs: batch_size, tgt_seq_len, embedding_size
		# hidden (h, c): 1, batch_size, en_hidden_size
		batch_size, tgt_seq_len = xs.shape[:2]
		# batch_size, tgt_seq_len, 2*en_hidden_size
		attn_weights = torch.cat(hidden, 2).repeat(tgt_seq_len, 1, 1).transpose(0, 1)
		# batch_size, tgt_seq_len, src_seq_len
		attn_weights = self.attn(torch.cat([xs, attn_weights], -1))
		# batch_size, tgt_seq_len, src_seq_len
		idx = torch.arange(
			end=self.max_seq_len
			, dtype=torch.float
			, device=self.device
			).repeat(batch_size, tgt_seq_len, 1)
		x_lens = x_lens.unsqueeze(-1).unsqueeze(-1).repeat(1, tgt_seq_len, self.max_seq_len)
		mask = idx < x_lens.to(self.device)
		attn_weights[~mask] = float('-inf')
		# batch_size, tgt_seq_len, src_seq_len
		return F.softmax(attn_weights, dim=-1)