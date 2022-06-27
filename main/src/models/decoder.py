#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import torch
import torch.nn as nn
import torch.nn.functional as F
# private
from src.models.attention import LSTMRNNDecoderAttention

class AttBiLSTMRNNDecoder(nn.Module):
	"""Bidirectional RNN Decoder with Long Short Term Memory (LSTM) Unit and Attention"""
	def __init__(self, embedding_size, en_hidden_size, de_hidden_size
		, de_num_layers, dropout_rate, tgt_vocab_size, device):
		super(AttBiLSTMRNNDecoder, self).__init__()
		self.attn = LSTMRNNDecoderAttention(
			en_hidden_size=en_hidden_size
			, de_hidden_size=de_hidden_size
			, device=device)
		self.attn_combine = torch.nn.Linear(
			embedding_size + en_hidden_size, 
			de_hidden_size)
		self.lstm = nn.LSTM(
			input_size=embedding_size, 
			hidden_size=de_hidden_size, 
			num_layers=de_num_layers, 
			batch_first=True, 
			dropout=0, 
			bidirectional=False)
		self.dropout = nn.Dropout(dropout_rate)
		self.out = torch.nn.Linear(de_hidden_size, tgt_vocab_size)

	def forward(self, x, hidden, encoder_output, x_lens):
		# x: batch_size, 1, embedding_dim
		# hidden (h, c): 1, batch_size, num_layers*hidden_size
		# encoder_output: batch_size, src_seq_len, hidden_size
		# x_lens: batch_size
		# batch_size, 1, max_seq_len
		attn_w = self.attn(hidden, encoder_output, x_lens)
		# batch_size, 1, en_hidden_size
		context = attn_w.bmm(encoder_output)
		# batch_size, 1, de_hidden_size
		x = self.attn_combine(torch.cat((x, context), 2))
		x = F.relu(x)
		x, (h, c) = self.lstm(x, hidden)
		# batch_size, de_hidden_size
		x = self.dropout(x).squeeze(1)
		# batch_size, tgt_vocab_size
		x = F.log_softmax(self.out(x), dim=-1)
		return x, (h, c)