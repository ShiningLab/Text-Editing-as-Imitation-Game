#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# dependency
# built-in
import os


class Config():
	# config settings
	def __init__(self):
		self.seed = 0
		self.load_ckpt = False
		# aor for Arithmetic Operators Restoration
		# aes for Arithmetic Equation Simplification
		# aec for Arithmetic Equation Correction
		self.env = 'aor'
		self.N = 100 if self.env == 'aes' else 10
		self.L = 5
		self.D = 10000
		env_name = '{}_{}N_{}L_{}D'.format(self.env, self.N, self.L, self.D)
		# trajectory generator - lcs, levenshtein, self
		self.metric = 'self' if self.env == 'aes' else 'levenshtein'
		# model - base_seq2seq_lstm, seq2seq_lstm
		# base_lstm, lstm, lstm0, lstm1, lstm2
		self.model = 'lstm'
		# trajectory augmentation
		self.aug_trajectory = False
		# learning
		self.enable_scheduler = True
		# adaptive loss weight
		if self.env == 'aor':
			self.adapt_loss = False
		else:
			self.adapt_loss = True
		# I/O
		# current path
		self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
		self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
		# task path
		self.TASK_PATH = os.path.join(
			env_name, self.metric, str(self.aug_trajectory), str(self.adapt_loss), str(self.seed))
		# checkpoint
		self.CKPT_PATH = os.path.join(self.RESOURCE_PATH, 'ckpts')
		self.CKPT_SAVE_PATH = os.path.join(self.CKPT_PATH, self.TASK_PATH)
		os.makedirs(self.CKPT_SAVE_PATH, exist_ok=True)
		self.CKPT_SAVE_POINT = os.path.join(self.CKPT_SAVE_PATH, '{}.pt'.format(self.model))
		# log
		self.LOG_PATH = os.path.join(self.RESOURCE_PATH, 'log')
		self.LOG_SAVE_PATH = os.path.join(self.LOG_PATH, self.TASK_PATH)
		os.makedirs(self.LOG_SAVE_PATH, exist_ok=True)
		self.LOG_SAVE_POINT = os.path.join(self.LOG_SAVE_PATH, '{}.pkl'.format(self.model))
		# data
		self.BOS_TOKEN = '<s>'
		self.EOS_TOKEN = '[SEP]'
		self.PAD_TOKEN = '<pad>'