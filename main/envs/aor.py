#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# built-in
import os
# private
from envs.distance import Editops
from envs.base import TextEditEnv
from envs.utils import load_txt_to_list, parse_pos


class GameEnv(TextEditEnv):
	"""
    The game environment class for Arithmetic Operators Restoration (AOR)
	"""
	def __init__(self, metric='levenshtein'):
		super(GameEnv, self).__init__()
		self.env = 'aor'
        # only insert operation in AOR
		# 2 for [position, token]
		self.tgt_seq_len = 2
		self.operators = set(['+', '-', '*', '/', '=='])
		# trajectory generator
		self.traj_generator = Editops(metric=metric)

	def get_data(self, N, L, D):
		RAW_AE = os.path.join(self.RAW_DATA_PATH, 'arithmetic_equation')
		RAW_AOR = os.path.join(RAW_AE, self.env, '{}N'.format(N), '{}L'.format(L), '{}D'.format(D))
		raw_train_xs = load_txt_to_list(os.path.join(RAW_AOR, 'train_x.txt'))
		raw_train_ys = load_txt_to_list(os.path.join(RAW_AOR, 'train_y.txt'))
		raw_valid_xs = load_txt_to_list(os.path.join(RAW_AOR, 'val_x.txt'))
		raw_valid_ys = load_txt_to_list(os.path.join(RAW_AOR, 'val_y.txt'))
		raw_test_xs = load_txt_to_list(os.path.join(RAW_AOR, 'test_x.txt'))
		raw_test_ys = load_txt_to_list(os.path.join(RAW_AOR, 'test_y.txt'))
        # save data size
		self.data_size_dict = {}
		self.data_size_dict['train'] = len(raw_train_xs)
		self.data_size_dict['valid'] = len(raw_valid_xs)
		self.data_size_dict['test'] = len(raw_test_xs)
		return (raw_train_xs, raw_train_ys, raw_valid_xs, raw_valid_ys, raw_test_xs, raw_test_ys)
	
	def make(self, N=10, L=5, D=10000):
		self.numbers = set([str(n) for n in range(2, N+2)])
		data = self.get_data(N, L, D)
		self.data_dict = {}
		for mode, (xs, ys) in zip(['train', 'valid', 'test'], [(data[0], data[1]), (data[2], data[3]), (data[4], data[5])]):
			states, actions = map(list, zip(*[self.get_trajectories(x, y) for x, y in zip(xs, ys)]))
			self.data_dict[mode] = {}
			self.data_dict[mode]['xs'] = states
			self.data_dict[mode]['ys'] = actions
		self.max_seq_len = max([len(x) for xs in self.data_dict['train']['xs'] for x in xs])
		self.max_infer_step = max([len(_) for _ in self.data_dict['train']['xs']])

	def get_trajectories(self, x, y, do_copy=True):
		if isinstance(x, str):
			# white space tokenization
			x, y = x.split(), y.split()
        # to avoid mutable issue
		if do_copy:
			x, y = x.copy(), y.copy()
        # initialize placeholder
		xs = [x.copy()]
		ys_ = []
		editops = self.traj_generator.editops(x, y)
		c = 0
        # only insert operation in AOR
        # operation, index i, index j
		for _, i, j in editops: 
			i += c
			y_ = ['<pos_{}>'.format(i), y[j]]
			x.insert(i, y[j]) 
			c += 1
			xs.append(x.copy()) 
			ys_.append(y_)
		ys_.append([self.DONE] * self.tgt_seq_len)
		return xs, ys_

	def one_step_infer(self, state, action, do_copy=True):
        # to avoid mutable issue
		if do_copy:
			state = state.copy()
        # check if final action
		if action.count(self.DONE) == len(action):
			return state
        # position, target token
		pos, tk = action
		if pos.startswith('<pos_') and tk in self.operators: 
			pos = parse_pos(pos)
			state.insert(pos, tk)
		return state