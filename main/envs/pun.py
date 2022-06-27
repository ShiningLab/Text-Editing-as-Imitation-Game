#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# built-in
import os
# public
from tqdm import tqdm, trange
# private
from envs.distance import Editops
from envs.base import TextEditEnv
from envs.utils import load_txt_to_list, parse_pos


class GameEnv(TextEditEnv):
	"""
	docstring for GameEnv
	"""
	def __init__(self, metric='levenshtein'):
		super(GameEnv, self).__init__()
		self.env = 'pun'
		self.metric = metric
		# 2 for [position, token]
		self.tgt_seq_len = 2
		self.NORMAL_TOKEN = 'O'
		self.COMMA_TOKEN = 'COMMA'
		self.PERIOD_TOKEN = 'PERIOD'
		self.QUESTION_TOKEN = 'QUESTION'
		self.puns = set([self.PERIOD_TOKEN, self.COMMA_TOKEN, self.QUESTION_TOKEN])
		# trajectory generator
		self.traj_generator = Editops(metric=metric)

	def tokenize(self, raw_data, tokenizer):
		word_or_pun_list, mask_list = [], []
		for i in trange(len(raw_data)):
			word_or_pun = raw_data[i]
			if word_or_pun in self.puns:
				word_or_pun_list.append(word_or_pun)
				mask_list.append(1)
			else:
				tokens = tokenizer.tokenize(word_or_pun)
				for j in range(len(tokens) - 1):
					word_or_pun_list.append(tokens[j])
					mask_list.append(0)
				word_or_pun_list.append(tokens[-1])
				mask_list.append(1)
		return word_or_pun_list, mask_list

	def merge_word_pun(self, raw_data):
		# 'complicated\tO', 'thing\tCOMMA','the\tO'
		# complicated, thing, COMMA, the
		word_or_pun_list = []
		for word_pun_pair in raw_data:
			pair = word_pun_pair.split()
			if len(pair) > 1:
				word, pun = pair
				word_or_pun_list.append(word)
				if pun is not self.NORMAL_TOKEN:
					word_or_pun_list.append(pun)
		return word_or_pun_list

	def split(self, raw_data: list, tokenizer, max_seq_len: int) -> list:
		# punctuation restoration
		word_or_pun_list = self.merge_word_pun(raw_data)
		word_or_pun_list, mask_list = self.tokenize(word_or_pun_list, tokenizer)
		# split sequence given max_seq_len
		initial_states_list, initial_masks_list, goal_states_list = [], [], []
		initial_state, initial_mask, goal_state = [], [], []
		x, x_m, y = [], [], []
		for word_or_pun, mask in zip(word_or_pun_list, mask_list):
			if word_or_pun not in self.puns:
				x.append(word_or_pun)
				x_m.append(mask)
			y.append(word_or_pun)
			if mask:
				if len(goal_state) + len(y) > max_seq_len - 1:
					initial_states_list.append(initial_state)
					initial_masks_list.append(initial_mask)
					if goal_state[0] in self.puns:
						goal_states_list[-1].append(goal_state[0])
						del goal_state[0]
					goal_states_list.append(goal_state)
					initial_state, initial_mask, goal_state = [], [], []
				initial_state += x
				initial_mask += x_m
				goal_state += y
				x, x_m, y = [], [], []
		if goal_state:
			initial_states_list.append(initial_state)
			initial_masks_list.append(initial_mask)
			goal_states_list.append(goal_state)
		return initial_states_list, initial_masks_list, goal_states_list

	def get_data(self, tokenizer, max_seq_len=None):
		RAW_PUN = os.path.join(self.RAW_DATA_PATH, 'punctuation_restoration')
		self.raw_train_data = load_txt_to_list(os.path.join(RAW_PUN, 'train2012.txt'))
		self.raw_valid_data = load_txt_to_list(os.path.join(RAW_PUN, 'dev2012.txt'))
		self.raw_test_data = load_txt_to_list(os.path.join(RAW_PUN, 'test2011.txt'))
		
		raw_train_data= self.split(self.raw_train_data, tokenizer, max_seq_len)
		raw_valid_data = self.split(self.raw_valid_data, tokenizer, max_seq_len)
		raw_test_data= self.split(self.raw_test_data, tokenizer, max_seq_len)
		
		return (raw_train_data, raw_valid_data, raw_test_data)
	
	def get_trajectories(self, x, x_m, y, do_copy=True):
		if do_copy:
			x, x_m, y = x.copy(), x_m.copy(), y.copy()
		xs = [x.copy()]
		x_masks = [x_m.copy()]
		ys_ = []
		editops = self.traj_generator.editops(x, y)
		c = 0 
		for _, i, j in editops: 
			i += c
			y_ = ['<pos_{}>'.format(i), y[j]]
			x.insert(i, y[j])
			x_m.insert(i, 1)
			c += 1
			xs.append(x.copy()) 
			x_masks.append(x_m.copy())
			ys_.append(y_)
		ys_.append([self.DONE] * self.tgt_seq_len)
		return xs, x_masks, ys_

	def make(self, tokenizer, max_seq_len):
		print('Loading...')
		data = self.get_data(tokenizer=tokenizer, max_seq_len=max_seq_len)
		self.data_dict = {}
		self.max_seq_len, self.max_infer_step = float('-inf'), float('-inf')
		for mode, d in zip(['train', 'valid', 'test'], data):
			print(' {} trajectories...'.format(mode))
			trajectories = []
			for x, x_m, y in zip(tqdm(d[0]), d[1], d[2]):
				trajectories.append(self.get_trajectories(x, x_m, y))
			states, state_masks, actions = map(list, zip(*trajectories))
			self.data_dict[mode] = {}
			self.data_dict[mode]['xs'] = states
			self.data_dict[mode]['x_masks'] = state_masks
			self.data_dict[mode]['ys'] = actions
			self.max_seq_len = max(self.max_seq_len, max([len(_) for xs in self.data_dict[mode]['xs'] for _ in xs]))
			self.max_infer_step = max(self.max_infer_step, max(len(_) for _ in self.data_dict[mode]['xs']))
	
	def one_step_infer(self, state, state_mask, action, do_copy=True):
		if do_copy:
			state, state_mask = state.copy(), state_mask.copy()
		if action.count(self.DONE) == len(action):
			return state, state_mask
		pos, tk = action
		if pos.startswith('<pos_') and tk in self.puns: 
			pos = parse_pos(pos)
			if pos in range(1, len(state)):
				if state[pos] not in self.puns and state[pos-1] not in self.puns:
					state.insert(pos, tk)
					state_mask.insert(pos, 1)
			elif pos == 0:
				pass
			# 	if state[pos] not in self.puns:
			# 		state.insert(pos, tk)
			# 		state_mask.insert(pos, 1)
			else:
				if state[-1] not in self.puns:
					state.insert(pos, tk)
					state_mask.insert(pos, 1)
		return state, state_mask