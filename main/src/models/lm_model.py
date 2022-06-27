#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import math
# public
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
# private
from src.models.lm import LAN_MODELS
from src.models.encoder import PositionalEncoding


class ModelGraph(nn.Module):
	"""docstring for ModelGraph"""
	def __init__(self, config, **kwargs):
		super(ModelGraph, self).__init__()
		self.config = config
		self.update_config(**kwargs)
		# embedding layer
		self.de_embedding_layer = nn.Embedding(
			num_embeddings=self.config.tgt_vocab_size
			, embedding_dim=self.config.de_embedding_size)
		self.de_position_embedding_layer = PositionalEncoding(
			d_model=self.config.de_embedding_size
			, dropout=self.config.dropout_rate)
		# language model
		if config.load_ckpt:
			print(' load lm from local...')
			pretrained_model_name_or_path = self.config.LM_CKPT_SAVE_PATH
		else:
			print(' load lm from huggingface...')
			pretrained_model_name_or_path = self.config.lm
		self.lm_layer = LAN_MODELS[self.config.lm].from_pretrained(
			pretrained_model_name_or_path
			, mirror=config.lm_mirror
			)
		# self attention layer
		decoder0_layer = nn.TransformerEncoderLayer(
				d_model=self.config.de_hidden_size
				, nhead=config.de_num_heads
				, dim_feedforward=config.de_ffd_size
				, dropout=config.dropout_rate
				, activation= 'gelu'
				, batch_first=True
				)
		self.decoder0 = nn.TransformerEncoder(
				encoder_layer=decoder0_layer
				, num_layers=self.config.de_num_layers
				)
		# cross attention layer
		decoder1_layer = nn.TransformerDecoderLayer(
				d_model=self.config.de_hidden_size
				, nhead=config.de_num_heads
				, dim_feedforward=config.de_ffd_size
				, dropout=config.dropout_rate
				, activation= 'gelu'
				, batch_first=True
				)
		self.decoder1 = nn.TransformerDecoder(
				decoder_layer=decoder1_layer
				, num_layers=self.config.de_num_layers
				)
		self.decoders = [self.decoder0, self.decoder1]
		self.out_layer0 = nn.Linear(
			in_features=self.config.de_hidden_size
			, out_features=self.config.tgt_vocab_size)
		self.out_layer1 = nn.Linear(
			in_features=self.config.de_hidden_size
			, out_features=self.config.tgt_vocab_size)
		self.out_layers = [self.out_layer0, self.out_layer1]
		# projection layer
		self.proj_layer = nn.Linear(
			in_features=self.config.src_seq_len
			, out_features=self.config.tgt_seq_len)
		# others
		self.dropout = nn.Dropout(self.config.dropout_rate)

	def update_config(self, **kwargs):
		# embedding
		self.config.de_embedding_size = self.config.lm_config.hidden_size
		# language model
		self.config.lm_hidden_size = self.config.lm_config.hidden_size
		# decoder
		self.config.de_num_layers = 3
		self.config.de_num_heads = 8
		self.config.de_hidden_size = self.config.lm_config.hidden_size
		self.config.de_ffd_size = 2048
		# others
		self.config.dropout_rate = 0.5
		self.config.teacher_forcing_ratio = 0.5
		# update configuration accordingly
		for k,v in kwargs.items():
			setattr(self.config, k, v)

	def decode(self, tgt, idx, memory=None, memory_key_padding_mask=None):
		if idx:
			# batch_size, tgt_seq_len, hidden_size
			xs = self.decoders[idx](tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
		else:
			# batch_size, tgt_seq_len, hidden_size
			xs = self.decoders[idx](tgt)
		# batch_size, tgt_seq_len, tgt_vocab_size
		return self.out_layers[idx](xs)

	def forward(self, xs, x_masks, ys=None):
		# xs: batch_size, src_seq_len
		# x_masks: batch_size, src_seq_len
		batch_size = xs.shape[0]
		# batch_size, src_seq_len, hidden_size
		encoder_output = self.dropout(self.lm_layer(xs, attention_mask=x_masks).last_hidden_state)
		# batch_size, tgt_seq_len, hidden_size
		xs = self.dropout(self.proj_layer(encoder_output.transpose(1, 2))).transpose(1, 2)
		# decoder 1
		# batch_size, tgt_seq_len, tgt_vocab_size
		ys0 = self.decode(xs, 0)
		# teacher forcing
		# batch_size, tgt_seq_len
		xs = torch.argmax(ys0, dim=-1)
		if self.training and ys is not None:
			mask = torch.rand(xs.shape) < self.config.teacher_forcing_ratio
			xs[mask] = ys[mask]
		# decoder 2
		# BOS: batch_size, 1
		bos = torch.empty((batch_size, 1), dtype=torch.int64, device=self.config.device)
		bos.fill_(self.config.DE_BOS_IDX)
		# batch_size, tgt_seq_len
		xs = torch.hstack([bos, xs[:, :-1]])
		# batch_size, tgt_seq_len, hidden_size
		xs = self.de_embedding_layer(xs) * math.sqrt(self.config.de_embedding_size)
		# batch_size, tgt_seq_len, hidden_size
		xs = self.de_position_embedding_layer(xs)
		# batch_size, tgt_seq_len, tgt_vocab_size
		ys1 = self.decode(xs, 1, encoder_output, x_masks)
		return F.log_softmax(ys0, dim=-1) + F.log_softmax(ys1, dim=-1)