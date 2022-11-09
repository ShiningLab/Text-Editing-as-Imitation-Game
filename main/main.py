#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# dependency
# built-in
import os
import random
# public
import torch
import numpy as np
# private
from config import Config
from src.utils import helper


class Agent(object):
	"""docstring for Agent"""
	def __init__(self):
		super(Agent, self).__init__()
		self.config = Config()
		self.initialize()
		# setup game environment
		self.get_env()
		# prepare vocabulary
		self.get_vocab()

	def set_seed(self):
		random.seed(self.config.seed)
		np.random.seed(self.config.seed)
		torch.manual_seed(self.config.seed)
		torch.cuda.manual_seed_all(self.config.seed)

	def initialize(self):
		# setup random seed
		self.set_seed()
		# setup device
		self.config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

	def get_env(self):
		print('Initializing Game Environment ...')
		self.env = helper.get_env(self.config)
		# self.env.set_seed(self.config.seed)
		if self.config.env in ['aor', 'aes', 'aec']:
			print('Constructing Editops with Metric %s ...' % self.config.metric)
			self.env.make(N=self.config.N, L=self.config.L, D=self.config.D)
			self.config.max_seq_len = self.env.max_seq_len
		else:
			raise NotImplementedError
		self.config.ori_train_size = len(self.env.data_dict['train']['xs'])
		self.config.valid_size = len(self.env.data_dict['valid']['xs'])
		self.config.test_size = len(self.env.data_dict['test']['xs'])
		self.config.max_infer_step = self.env.max_infer_step
		self.config.tgt_seq_len = self.env.tgt_seq_len

	def get_vocab(self):
		self.env.vocab_dict = helper.get_vocab(self.env, self.config)
		self.config.tgt_vocab_size = len(self.env.vocab_dict['tgt_token2idx'])
		self.config.src_vocab_size = len(self.env.vocab_dict['src_token2idx'])
		self.config.PAD_IDX = self.env.vocab_dict['src_token2idx'][self.config.PAD_TOKEN]
		self.config.BOS_IDX = self.env.vocab_dict['tgt_token2idx'].get(self.config.BOS_TOKEN)

	def get_model(self):
		self.model = helper.get_model(self.config).to(self.config.device)
		self.config.num_parameters = helper.count_parameters(self.model)

	def learn(self):
		# setup model
		self.get_model()
		# initialize trainer
		trainer_class = helper.get_trainer(self.config)
		trainer = trainer_class.Trainer(
			model=self.model
			, env=self.env
			, config=self.config
			)
		# ready to go
		trainer.train()
		# trainer.eva()


def main(): 
	a = Agent()
	a.learn()

if __name__ == '__main__':
	  main()