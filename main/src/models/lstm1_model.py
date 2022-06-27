#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# public
import torch
import torch.nn as nn
import torch.nn.functional as F
# private
from src.utils import helper
from src.models.encoder import BiLSTMRNNEncoder


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
		# encoder
		self.encoder = BiLSTMRNNEncoder(
			embedding_dim=self.config.en_embedding_size
			, hidden_size=self.config.en_hidden_size//2
			, num_layers=self.config.en_num_layers
			, max_seq_len=self.config.max_seq_len
			, dropout_rate=self.config.dropout_rate
			, proj=True)
		# projection layer
		self.proj_layer = nn.Linear(
			in_features=self.config.max_seq_len*self.config.en_hidden_size
			, out_features=self.config.tgt_seq_len*self.config.tgt_vocab_size)
		# others
		self.dropout = nn.Dropout(self.config.dropout_rate)
		# parameters initialization
		self.init_parameters()
	
	def update_config(self, **kwargs):
		# encoder
		self.config.en_embedding_size = 300 if self.config.embed else 512
		self.config.en_hidden_size = 512
		# 4 for ablation study
		# 6 for a fair parameter size
		self.config.en_num_layers = 4
		# others
		self.config.hidden_size = 512
		self.config.dropout_rate = 0.5
		# update configuration accordingly
		for k,v in kwargs.items():
			setattr(self.config, k, v)
	
	def init_parameters(self): 
		for name, parameters in self.named_parameters():
			if 'weight' in name:
				torch.nn.init.normal_(parameters.data, mean=0, std=0.01)
			else:
				torch.nn.init.constant_(parameters.data, 0)
				
	def forward(self, xs, x_lens, ys=None):
		# xs: batch_size, src_seq_len
		# x_lens: batch_size
		batch_size = xs.shape[0]
		# encoder
		# batch_size, src_seq_len, embedding_size
		xs = self.dropout(self.en_embedding_layer(xs))
		# xs: batch_size, src_seq_len, en_hidden_size
		# hidden (h, c): 1, batch_size, en_hidden_size
		xs, _ = self.encoder(xs, x_lens)
		# decoder 1
		# batch size, tgt_seq_len, tgt_vocab_size
		ys = self.proj_layer(xs.reshape(batch_size, -1)).reshape(batch_size, self.config.tgt_seq_len, -1)
		
		return F.log_softmax(ys, dim=-1)