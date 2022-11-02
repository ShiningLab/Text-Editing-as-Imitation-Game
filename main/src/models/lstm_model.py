#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# built in
import copy
# public
import torch
import torch.nn as nn
import torch.nn.functional as F
# private
from src.utils import helper
from src.models.encoder import BiLSTMRNNEncoder
from src.models.attention import SelfDecoderAttention


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
		# encoder
		self.encoder = BiLSTMRNNEncoder(
			embedding_dim=self.config.en_embedding_size
			, hidden_size=self.config.en_hidden_size//2
			, num_layers=self.config.en_num_layers
			, max_seq_len=self.config.max_seq_len
			, dropout_rate=self.config.dropout_rate
			, proj=True)
		# attention
		self.attn = SelfDecoderAttention(
			en_hidden_size=self.config.en_hidden_size
			, de_embedding_size=self.config.de_embedding_size
			, de_hidden_size=self.config.de_hidden_size
			, max_seq_len=self.config.max_seq_len
			, device=self.config.device)
		self.attn_combine = nn.Linear(
			in_features=self.config.de_embedding_size + self.config.en_hidden_size
			, out_features=self.config.de_hidden_size)
		# decoder
		self.decoder0 = nn.LSTM(
			input_size=self.config.de_embedding_size
			, hidden_size=self.config.de_hidden_size
			, num_layers=self.config.de_num_layers
			, batch_first=True
			, bidirectional=False)
		self.decoder1 = copy.deepcopy(self.decoder0)
		self.decoders = [self.decoder0, self.decoder1]
		self.out_layer0 = nn.Linear(
			in_features=self.config.de_hidden_size
			, out_features=self.config.tgt_vocab_size)
		self.out_layer1 = copy.deepcopy(self.out_layer0)
		self.out_layers = [self.out_layer0, self.out_layer1]
		# projection layer
		self.proj_layer = nn.Linear(
			in_features=self.config.max_seq_len
			, out_features=self.config.tgt_seq_len)
		# others
		self.dropout = nn.Dropout(self.config.dropout_rate)
		# parameters initialization
		self.init_parameters()
	
	def update_config(self, **kwargs):
		# encoder
		self.config.en_embedding_size = 512
		self.config.en_hidden_size = 512
		self.config.en_num_layers = 4
		# decoder
		self.config.de_embedding_size = 512
		self.config.de_hidden_size = 512
		self.config.de_num_layers = 1
		# others
		self.config.dropout_rate = 0.5
		self.config.teacher_forcing_ratio = 0.5
		# update configuration accordingly
		for k,v in kwargs.items():
			setattr(self.config, k, v)
	
	def init_parameters(self): 
		for name, parameters in self.named_parameters():
			if 'weight' in name:
				torch.nn.init.normal_(parameters.data, mean=0, std=0.01)
			else:
				torch.nn.init.constant_(parameters.data, 0)

	def decode(self, xs, x_lens, encoder_output, hidden, idx):
		# batch_size, tgt_seq_len, src_seq_len
		attn_w = self.attn(xs, x_lens, hidden)
		# batch_size, tgt_seq_len, en_hidden_size
		attn_applied = attn_w.bmm(encoder_output)
		# batch_size, tgt_seq_len, de_hidden_size
		xs = F.relu(self.attn_combine(torch.cat([xs, attn_applied], -1)))
		# batch_size, tgt_seq_len, de_hidden_size
		xs, hidden = self.decoders[idx](xs, hidden)
		# xs, hidden = self.decoders[idx](xs, hidden)
		return self.out_layers[idx](self.dropout(xs)), hidden

	def forward(self, xs, x_lens, ys=None):
		# xs: batch_size, src_seq_len
		# x_lens: batch_size
		batch_size = xs.shape[0]
		#encoder
		# batch_size, src_seq_len, embedding_size
		xs = self.dropout(self.en_embedding_layer(xs))
		# encoder_output: batch_size, src_seq_len, en_hidden_size
		# hidden (h, c): 1, batch_size, en_hidden_size
		encoder_output, encoder_hidden = self.encoder(xs, x_lens)
		# batch_size, tgt_seq_len, en_hidden_size
		xs = self.dropout(self.proj_layer(encoder_output.transpose(1, 2))).transpose(1, 2)
		# decoder 1
		# batch_size, tgt_seq_len, tgt_vocab_size
		ys0, decoder_hidden = self.decode(xs, x_lens, encoder_output, encoder_hidden, 0)
		# teacher forcing
		xs = torch.argmax(ys0, dim=-1)
		if self.training and ys is not None:
			mask = torch.rand(xs.shape) < self.config.teacher_forcing_ratio
			xs[mask] = ys[mask]
		# decoder 2
		# BOS: batch_size, 1
		bos = torch.empty((batch_size, 1), dtype=torch.int64, device=self.config.device)
		bos.fill_(self.config.BOS_IDX)
		# batch_size, tgt_seq_len
		xs = torch.hstack([bos, xs[:, :-1]])
		# batch_size, tgt_seq_len, embedding_size
		xs = self.dropout(self.de_embedding_layer(xs))
		# batch_size, tgt_seq_len, tgt_vocab_size
		ys1, decoder_hidden = self.decode(xs, x_lens, encoder_output, decoder_hidden, 1)
		return F.log_softmax(ys0, dim=-1) + F.log_softmax(ys1, dim=-1)