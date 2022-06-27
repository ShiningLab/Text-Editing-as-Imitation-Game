#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os
# public

# private
from envs.base import TextEditEnv
from envs.utils import load_txt_to_list, get_levenshtein_editops


class GameEnv(TextEditEnv):
	"""
	docstring for GameEnv
	"""
	def __init__(self):
		super(GameEnv, self).__init__()
		self.tgt_seq_len = 3
		
	def preprocess(self, raw_sents, tokenizer=None):
		if not tokenizer:
			# lower case and white tokenization
			sents = [s.lower().split() for s in raw_sents]
		else:
			sents = [tokenizer.tokenize(s) for s in raw_sents]
		# remove tokens within parentheses
		# sents = [self.replace_parentheses(s) for s in sents]
		return sents

	def get_data(self, tokenizer=None):
		# load data
		RAW_PATH = os.path.join(self.RAW_DATA_PATH, 'wikismall')
		data = []
		for f in ['PWKP_108016.tag.80.aner.ori.train.src', 'PWKP_108016.tag.80.aner.ori.train.dst'
		, 'PWKP_108016.tag.80.aner.ori.valid.src', 'PWKP_108016.tag.80.aner.ori.valid.dst'
		, 'PWKP_108016.tag.80.aner.ori.test.src', 'PWKP_108016.tag.80.aner.ori.test.dst']:
			d = load_txt_to_list(os.path.join(RAW_PATH, f))
			d = self.preprocess(d, tokenizer=tokenizer)
			data.append(d)
		self.max_seq_len = max([len(x) for x in data[0]])
		return data

	def get_trajectories(self, x: str, y: str) -> list:
		if isinstance(x, str):
			# white space tokenization
			x, y = x.split(), y.split()
		xs = [x.copy()]
		ys_= []
		editops = get_levenshtein_editops(x, y)
		c = 0
		for tag, i, j in editops:
			i += c
			if tag == 'replace':
				y_ = ['<sub>', '<pos_{}>'.format(i), y[j]]
				x[i] = y[j]
			elif tag == 'delete':
				y_ = ['<delete>', '<pos_{}>'.format(i), '<pos_{}>'.format(i)]
				del x[i]
				c -= 1
			elif tag == 'insert': 
				y_ = ['<insert>', '<pos_{}>'.format(i), y[j]]
				x.insert(i, y[j]) 
				c += 1
			else:
				raise NotImplementedError
			xs.append(x.copy()) 
			ys_.append(y_)
		ys_.append(['<done>']*self.tgt_seq_len)
		return xs, ys_

	def make(self, tokenizer=None):
		data = self.get_data(tokenizer=tokenizer)
		self.data_dict = {}
		for mode, (xs, ys) in zip(['train', 'valid', 'test'], [(data[0], data[1]), (data[2], data[3]), (data[4], data[5])]):
			states, actions = map(list, zip(*[self.get_trajectories(x, y) for x, y in zip(xs, ys)]))
			self.data_dict[mode] = {}
			self.data_dict[mode]['xs'] = states
			self.data_dict[mode]['ys'] = actions
		self.max_infer_step = max(len(_) for _ in self.data_dict['train']['xs'])