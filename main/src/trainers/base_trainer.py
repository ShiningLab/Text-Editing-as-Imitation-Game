#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# public
import torch
from torch.utils import data as torch_data
# private
from src.utils import helper


class Base_Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, model, env, config, tokenizer=None):
		super().__init__()
		self.model = model
		self.env = env
		self.config = config
		self.tokenizer = tokenizer
		self.__update_config()
		self.__initialize()
		self.setup_log_dict()

	def __update_config(self):
		# data
		self.config.eva = False
		self.config.sampling = True
		self.config.shuffle = True
		self.config.num_workers = 0
		self.config.pin_memory = True
		self.config.drop_last = True
		# train
		self.config.batch_size = 256
		self.config.learning_rate = 5e-4
		self.config.valid_patience = 512
		self.config.clipping_threshold = 5.

	def __initialize(self):
		self.step, self.epoch = 0, 0
		self.done = False
		self.dataloader_dict = {}
		self.valid_epoch = 0

	def setup_log_dict(self):
		self.log_dict = {}
		for mode in ['train', 'valid', 'test', 'eva']:
			self.log_dict[mode] = {}
		for mode in ['valid', 'best_valid', 'test']:
			self.log_dict['eva'][mode] = {}
		for mode in ['train', 'valid', 'test']:
			for k in ['loss', 'token_acc', 'seq_acc']:
				self.log_dict[mode][k] = []
		self.log_dict['eva']['best_valid']['valid_key_metric'] = float('-inf')

	def load_ckpt(self):
		ckpt_to_load =  torch.load(self.config.CKPT_SAVE_POINT, map_location=self.config.device) 
		self.step = ckpt_to_load['step'] 
		self.epoch = ckpt_to_load['epoch'] 
		self.log_dict = ckpt_to_load['log_dict']
		self.optimizer.load_state_dict(ckpt_to_load['optimizer'])
		model_state_dict = ckpt_to_load['model']
		self.model.load_state_dict(model_state_dict) 
		print('Model restored from {}.'.format(self.config.CKPT_SAVE_POINT))