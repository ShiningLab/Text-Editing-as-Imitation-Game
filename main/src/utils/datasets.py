#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# dependency
# built-in
import random
# public
import numpy as np
from tqdm import tqdm
from torch.utils import data as torch_data


class AEDataset(torch_data.Dataset):
	"""The class of dataset for game Arithmetic Equation"""
	def __init__(self, env, augmentor, mode, config):
		super(AEDataset, self).__init__()
		self.env, self.augmentor, self.mode, self.config = env, augmentor, mode, config
		self.xs, self.ys = env.data_dict[mode]['xs'], env.data_dict[mode]['ys']
		self.data_size = len(self.xs)
		if config.aug_trajectory and mode == 'train':
			self.do_augmentation()

	def do_augmentation(self):
		# trajectory augmentation
		print('Augmenting Trajectories as Initial States ...')
		aug_xs, aug_ys = self.augmentor(self.xs, self.ys, self.env)
		self.xs += aug_xs
		self.ys += aug_ys
		self.data_size = len(self.xs)
		print('Data Size {} -> {} by Trajectory Augmentation.'.format(
			self.data_size - len(aug_xs), self.data_size))

	def __len__(self): 
		return self.data_size
	
	def __getitem__(self, idx):
		return self.xs[idx], self.ys[idx]