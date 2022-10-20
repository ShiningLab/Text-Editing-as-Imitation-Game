#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os
# public
from transformers import AutoConfig
# private
from res.settings import SPECIAL_TOKENS_DICT


class Config():
	# config settings
	def __init__(self):
		self.seed = 0
		self.load_ckpt = False
		# aor for Arithmetic Operators Restoration
		# aes for Arithmetic Equation Simplification
		# aec for Arithmetic Equation Correction
		# pun for Punctuation Restoration
		self.env = 'pun'
		if self.env in ['aor', 'aes', 'aec']:
			self.N = 10
			self.L = 5
			self.D = 10000
			env_name = '{}_{}N_{}L_{}D'.format(self.env, self.N, self.L, self.D)
		else:
			# pun
			self.max_seq_len = 64
			env_name = self.env
		# trajectory generator - lcs, levenshtein, damerau-levenshtein, self
		self.metric = 'levenshtein'
		# model - base_seq2seq_lstm, seq2seq_lstm
		# base_lstm, lstm, lstm0
		# lm, lm_lstm
		self.model = 'lm_lstm'
		# trajectory augmentation
		self.aug_trajectory = True
		# learning
		self.enable_scheduler = True
		# adaptive loss weight
		if self.env == 'aor':
			self.adapt_loss = False
		else:
			self.adapt_loss = True
		# None, glove
		self.embed = None
		# None, bert-base-uncased
		self.lm = 'bert-base-uncased'
		if self.lm:
			self.lm_mirror = 'https://mirrors.pku.edu.cn/hugging-face-models'
			self.lm_config = AutoConfig.from_pretrained(self.lm)
		# check
		# language model can not be None for real tasks
		if not self.lm and self.env not in ['aor', 'aes', 'aec']: 
			raise NotImplementedError
		# I/O
		# current path
		self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
		self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
		# task path
		self.TASK_PATH = os.path.join(
			env_name, self.metric, str(self.aug_trajectory)
			, str(self.adapt_loss), str(self.lm).replace('/', '-'), str(self.seed))
		# checkpoint
		self.CKPT_PATH = os.path.join(self.RESOURCE_PATH, 'ckpts')
		self.CKPT_SAVE_PATH = os.path.join(self.CKPT_PATH, self.TASK_PATH)
		self.LM_CKPT_SAVE_PATH = os.path.join(self.CKPT_SAVE_PATH, str(self.lm).replace('/', '-'))
		os.makedirs(self.LM_CKPT_SAVE_PATH, exist_ok=True)
		self.CKPT_SAVE_POINT = os.path.join(self.CKPT_SAVE_PATH, '{}.pt'.format(self.model))
		# log
		self.LOG_PATH = os.path.join(self.RESOURCE_PATH, 'log')
		self.LOG_SAVE_PATH = os.path.join(self.LOG_PATH, self.TASK_PATH)
		os.makedirs(self.LOG_SAVE_PATH, exist_ok=True)
		self.LOG_SAVE_POINT = os.path.join(self.LOG_SAVE_PATH, '{}.pkl'.format(self.model))
		# data
		self.BOS_TOKEN = SPECIAL_TOKENS_DICT[self.lm]['bos_token']
		self.EOS_TOKEN = SPECIAL_TOKENS_DICT[self.lm]['eos_token']
		self.PAD_TOKEN = SPECIAL_TOKENS_DICT[self.lm]['pad_token']
		# pre-trained embeddings
		if self.embed:
			self.embed_path = os.path.join(self.RESOURCE_PATH, 'embeddings', '{}.pkl'.format(self.embed))