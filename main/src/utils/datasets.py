#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import random
# public
import numpy as np
from tqdm import tqdm
from torch.utils import data as torch_data


# dataset for game Arithmetic Equation
class AEDataset(torch_data.Dataset):
	"""docstring for AEDataset"""
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


# dataset for game Punctuation Restoration
class PunDataset(torch_data.Dataset):
	"""docstring for PunDataset"""
	def __init__(self, env, augmentor, mode, config):
		super(PunDataset, self).__init__()
		self.switch = False
		self.env, self.augmentor, self.mode, self.config = env, augmentor, mode, config
		if mode == 'train':
			# enable sequence boundary sampling
			# adapted from https://arxiv.org/pdf/2106.06731.pdf
			self.raw_init_states = [_ for x in  env.data_dict['train']['xs'] for _ in x[0]]
			self.raw_init_masks = [_ for x_m in  env.data_dict['train']['x_masks'] for _ in x_m[0]]
			self.raw_goal_states = [_ for x in  env.data_dict['train']['xs'] for _ in x[-1]]
			self.get_goal2init_dict()
			self.get_sample_pool(
				size=len(self.raw_goal_states)-(env.max_seq_len-1)
				, invalids=set([i for i, token in enumerate(self.raw_goal_states) if token in env.puns])
				)
			self.data_size = len(env.data_dict[mode]['xs'])
			# to record trajectories
			self.states_list, self.state_masks_list, self.actions_list = [], [], []
		else:
			self.xs, self.x_ms, self.ys = env.data_dict[mode]['xs'], env.data_dict[mode]['x_masks'], env.data_dict[mode]['ys']
			self.data_size = len(self.xs)

	def get_goal2init_dict(self):
		self.goal2init = dict()
		i = 0
		for j in range(len(self.raw_goal_states)):
			self.goal2init[j] = i
			if self.raw_goal_states[j] in self.env.puns:
				i -= 1
			i += 1
		self.max_goal_idx = max(self.goal2init.keys())

	def get_sample_pool(self, size, invalids=None):
		self.sample_pool = list(range(size))
		if invalids:
			self.sample_pool = [i for i in self.sample_pool if i not in invalids]
		random.shuffle(self.sample_pool)

	def do_aug(self, states, state_masks, actions):
		aug_state, aug_state_mask, aug_action = self.augmentor(states, state_masks, self.env)
		states += aug_state
		state_masks += aug_state_mask
		actions += aug_action
		return states, state_masks, actions

	def __len__(self): 
		return self.data_size
	
	def __getitem__(self, idx):
		if self.mode == 'train':
			if self.switch:
				if not self.sample_pool:
					self.get_sample_pool(len(self.states_list))
				idx = self.sample_pool.pop()
				states, state_masks, actions = self.states_list[idx], self.state_masks_list[idx], self.actions_list[idx]
				if self.config.aug_trajectory:
					return self.do_aug(states, state_masks, actions)
				else:
					return states, state_masks, actions
			else:
				goal_si = self.sample_pool.pop()
				goal_ei = goal_si + (self.env.max_seq_len-1)
				init_si, init_ei = self.goal2init[goal_si], self.goal2init[goal_ei]
				# edge case
				if self.raw_goal_states[goal_ei] in self.env.puns:
					goal_ei += 1
				goal_state = self.raw_goal_states[goal_si:goal_ei]
				init_state = self.raw_init_states[init_si:init_ei]
				init_mask = self.raw_init_masks[init_si:init_ei]
				states, state_masks, actions = self.env.get_trajectories(init_state, init_mask, goal_state)
				self.states_list.append(states)
				self.state_masks_list.append(state_masks)
				self.actions_list.append(actions)
				if not self.sample_pool:
					self.switch = True
				if self.config.aug_trajectory:
					return self.do_aug(states, state_masks, actions)
				else:
					return states, state_masks, actions
		else:
			return self.xs[idx], self.x_ms[idx], self.ys[idx]