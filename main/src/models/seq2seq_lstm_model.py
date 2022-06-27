#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import random
# public
import torch
import torch.nn as nn
import torch.nn.functional as F
# private
from src.utils import helper
from src.models.encoder import BiLSTMRNNEncoder
from src.models.decoder import AttBiLSTMRNNDecoder


class ModelGraph(nn.Module):
	"""docstring for ModelGraph"""
	def __init__(self, config, **kwargs):
		super(ModelGraph, self).__init__()
		self.config = config
		self.update_config(**kwargs)
		# embedding layer
		self.en_embedding_layer = nn.Embedding(
			num_embeddings=self.config.src_vocab_size
			, embedding_dim=self.config.en_embedding_size
			, padding_idx=self.config.PAD_IDX)
		self.de_embedding_layer = nn.Embedding(
			num_embeddings=self.config.tgt_vocab_size
			, embedding_dim=self.config.de_embedding_size)
		self.dropout = nn.Dropout(self.config.dropout_rate)
		# encoder
		self.encoder = BiLSTMRNNEncoder(
			embedding_dim=self.config.en_embedding_size
			, hidden_size=self.config.en_hidden_size // 2
			, num_layers=self.config.en_num_layers
			, max_seq_len=self.config.max_seq_len
			, dropout_rate=self.config.dropout_rate
			, proj=True)
		# decoder
		self.decoder = AttBiLSTMRNNDecoder(
			embedding_size=self.config.de_embedding_size
			, en_hidden_size=self.config.en_hidden_size
			, de_hidden_size=self.config.de_hidden_size
			, de_num_layers=self.config.de_num_layers
			, dropout_rate=self.config.dropout_rate
			, tgt_vocab_size=self.config.tgt_vocab_size
			, device=self.config.device)
		# parameters initialization
		self.init_parameters()

	def update_config(self, **kwargs):
		# encoder
		self.config.en_embedding_size = 300 if self.config.embed else 512
		self.config.en_hidden_size = 512
		self.config.en_num_layers = 4
		self.config.dropout_rate = 0.5
		# decoder
		self.config.de_embedding_size = 512
		self.config.de_hidden_size = 512
		self.config.de_num_layers = 1
		self.config.teacher_forcing_ratio = 0.5
		# update configuration accordingly
		for k, v in kwargs.items():
			setattr(self.config, k, v)

	def init_parameters(self): 
		for name, parameters in self.named_parameters():
			if 'weight' in name:
				torch.nn.init.normal_(parameters.data, mean=0, std=0.01)
			else:
				torch.nn.init.constant_(parameters.data, 0)
	
	def forward(self, xs, x_lens, ys=None):
		# xs: batch_size, max_seq_len
		# x_lens: batch_size
		# ys: batch_size, tgt_seq_len
		batch_size = xs.shape[0]
		# batch_size, src_seq_len, embedding_size
		encoder_output = self.dropout(self.en_embedding_layer(xs))
		# encoder_output: batch_size, src_seq_len, hidden_size
		# decoder_hidden (h, c): 1, batch_size, n_hidden_size
		encoder_output, decoder_hidden = self.encoder(encoder_output, x_lens)
		# batch_size
		decoder_input = torch.empty(
			batch_size, 
			dtype=torch.int64, 
			device=self.config.device)
		decoder_input.fill_(self.config.BOS_IDX)
		# tgt_seq_len, batch_size, tgt_vocab_size
		decoder_outputs = torch.zeros(
			self.config.tgt_seq_len, 
			batch_size, 
			self.config.tgt_vocab_size, 
			device=self.config.device)
		# greedy search with teacher forcing
		for i in range(self.config.tgt_seq_len):
			# batch_size, 1, embedding_size
			decoder_input = self.dropout(self.de_embedding_layer(decoder_input).unsqueeze(1))
			# decoder_output: batch_size, tgt_vocab_size
			# decoder_hidden (h, c): 1, batch_size, de_hidden_size
			decoder_output, decoder_hidden = self.decoder(
				decoder_input, decoder_hidden, encoder_output, x_lens)
			# batch_size, vocab_size
			decoder_outputs[i] = decoder_output
			# batch_size
			if self.training and ys is not None:
				decoder_input = ys[:, i] if random.random() < self.config.teacher_forcing_ratio \
				else decoder_output.max(1)[1]
			else:
				decoder_input = decoder_output.max(1)[1]
		# batch_size, max_ys_seq_len, vocab_size
		return decoder_outputs.transpose(0, 1)