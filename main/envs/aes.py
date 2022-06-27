#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# built-in
import os
# private
from envs.distance import Editops
from envs.base import TextEditEnv
from envs.utils import load_txt_to_list, parse_pos


class GameEnv(TextEditEnv):
	"""
	docstring for GameEnv
	"""
	def __init__(self, metric='self'):
		super(GameEnv, self).__init__()
		self.env = 'aes'
		self.metric = metric
		self.ops = set(['<replace>', '<delete>', '<insert>'])
		self.operators = set(['+', '-', '*', '/', '=='])
		# trajectory generator
		# 3 for [position, position, token]
		self.tgt_seq_len = 3
		if self.metric == 'self':
			self.get_trajectories = self.get_self_trajectories
			self.one_step_infer = self.one_step_self_infer
		else:
			self.traj_generator = Editops(metric=metric)
			self.get_trajectories = self.get_aes_trajectories
			self.one_step_infer = self.one_step_aes_infer

	def get_data(self, N, L, D):
		RAW_AE = os.path.join(self.RAW_DATA_PATH, 'arithmetic_equation')
		RAW_AOR = os.path.join(RAW_AE, self.env, '{}N'.format(N), '{}L'.format(L), '{}D'.format(D))
		raw_train_xs = load_txt_to_list(os.path.join(RAW_AOR, 'train_x.txt'))
		raw_train_ys = load_txt_to_list(os.path.join(RAW_AOR, 'train_y.txt'))
		raw_valid_xs = load_txt_to_list(os.path.join(RAW_AOR, 'val_x.txt'))
		raw_valid_ys = load_txt_to_list(os.path.join(RAW_AOR, 'val_y.txt'))
		raw_test_xs = load_txt_to_list(os.path.join(RAW_AOR, 'test_x.txt'))
		raw_test_ys = load_txt_to_list(os.path.join(RAW_AOR, 'test_y.txt'))

		self.data_size_dict = {}
		self.data_size_dict['train'] = len(raw_train_xs)
		self.data_size_dict['valid'] = len(raw_valid_xs)
		self.data_size_dict['test'] = len(raw_test_xs)

		return (raw_train_xs, raw_train_ys, raw_valid_xs, raw_valid_ys, raw_test_xs, raw_test_ys)
	
	def make(self, N=100, L=5, D=10000):
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

	def get_self_trajectories(self, x, y, do_copy=True):
		if isinstance(x, str):
			# white space tokenization
			x, y = x.split(), y.split()
		if do_copy:
			x, y = x.copy(), y.copy()
		xs = [x.copy()]
		ys_ = []
		num_left = len([i for i in x if i == '('])
		for i in range(num_left):
			left_idx = x.index('(') 
			right_idx = x.index(')') 
			v = y[left_idx] 
			# pos, pos, token
			ys_.append(['<pos_{}>'.format(left_idx), '<pos_{}>'.format(right_idx), v])
			# pos, token, pos
			# ys_.append(['<pos_{}>'.format(left_idx), v, '<pos_{}>'.format(right_idx)])
			# token, pos, pos
			# ys_.append([v, '<pos_{}>'.format(left_idx), '<pos_{}>'.format(right_idx)])
			x = x[:left_idx] + [v] + x[right_idx+1:]
			xs.append(x)
		ys_.append(['<done>']*3)
		return xs, ys_

	def get_aes_trajectories(self, x, y, do_copy=True):
		if isinstance(x, str):
			# white space tokenization
			x, y = x.split(), y.split()
		if do_copy:
			x, y = x.copy(), y.copy()
		xs = [x.copy()]
		ys_ = []
		editops = self.traj_generator.editops(x, y)
		c = 0
		for op, i, j in editops:
			i += c
			if op == 'replace':
				y_ = ['<replace>', '<pos_{}>'.format(i), y[j]]
				x[i] = y[j]
			elif op == 'delete':
				y_ = ['<delete>', '<pos_{}>'.format(i), '<pos_{}>'.format(i)]
				del x[i]
				c -= 1
			elif op == 'insert':
				y_ = ['<insert>', '<pos_{}>'.format(i), y[j]]
				x.insert(i, y[j])
				c += 1
			else:
				raise NotImplementedError
			xs.append(x.copy())
			ys_.append(y_)
		ys_.append([self.DONE] * self.tgt_seq_len)
		return xs, ys_

	def one_step_self_infer(self, state, action, do_copy=True):
		if do_copy:
			state = state.copy()
		if action.count(self.DONE) == len(action):
			return state
		# pos, pos, tk
		pos0, pos1, tk = action
		# pos, tk, pos
		# pos0, tk, pos1 = action
		# tk, pos, pos
		# tk, pos0, pos1 = action
		if pos0.startswith('<pos_') and pos1.startswith('<pos_')  and tk in self.numbers:
			pos0, pos1 = parse_pos(pos0), parse_pos(pos1)
			state = state[:pos0] + [tk] + state[pos1+1:]
		return state

	def one_step_aes_infer(self, state, action, do_copy=True):
		if do_copy:
			state = state.copy()
		if action.count(self.DONE) == len(action):
			return state
		op, pos, tk = action
		if op in self.ops and pos.startswith('<pos_'):
			pos = parse_pos(pos)
			if op == '<replace>' and pos in range(len(state)) and tk in self.numbers:
				state[pos] = tk
			elif op == '<delete>' and pos in range(len(state)):
				del state[pos]
			elif op == '<insert>' and pos in range(len(state)+1) and tk in self.numbers:
				state.insert(pos, tk)
			else:
				pass
		return state