#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# dependency
# built-in
# public
import numpy as np


class ActionEvaluater(object):
	"""docstring for ActionEvaluater"""
	def __init__(self, ys, ys_):
		super(ActionEvaluater, self).__init__()
		self.ys, self.ys_ = ys, ys_
		self.size = len(ys)
		self.token_acc = self.get_token_acc()
		self.seq_acc = self.get_seq_acc()

	def check_token(self, y, y_):
		min_len = min([len(y), len(y_)])
		return np.float32(sum(np.equal(np.array(y[:min_len], dtype=object), np.array(y_[:min_len], dtype=object))) / len(y))

	def check_seq(self, y, y_): 
		min_len = min([len(y), len(y_)]) 
		if sum(np.equal(np.array(y[:min_len], dtype=object), np.array(y_[:min_len]), dtype=object)) == len(y): 
			return 1
		return 0

	def get_token_acc(self):
		a = 0
		for i in range(self.size):
			y = self.ys[i]
			y_ = self.ys_[i]
			a += self.check_token(y, y_)
		return np.float32(a/self.size)

	def get_seq_acc(self): 
		a = 0
		for i in range(self.size):
			y = self.ys[i]
			y_ = self.ys_[i]
			a += self.check_seq(y, y_)
		return np.float32(a/self.size)


class Evaluater(object):
	"""docstring for Evaluater"""
	def __init__(self, ys, ys_, config):
		super(Evaluater, self).__init__()
		self.ys, self.ys_ = ys, ys_
		self.size = len(ys)
		self.token_acc = self.get_token_acc()
		self.seq_acc = self.get_seq_acc()
		self.eq_acc = self.get_eq_acc()
		self.key_metric = self.token_acc if config.env == 'aes' else self.eq_acc
		self.keys = ['token_acc', 'seq_acc', 'eq_acc']
		self.values = [self.token_acc, self.seq_acc, self.eq_acc]
		self.msg = ' Token Acc:{:.4f} Seq Acc:{:.4f} Eq Acc:{:.4f}'.format(self.token_acc, self.seq_acc, self.eq_acc)
		
	def check_token(self, y, y_):
		min_len = min([len(y), len(y_)])
		return np.float32(sum(np.equal(np.array(y[:min_len], dtype=object), np.array(y_[:min_len], dtype=object))) / len(y))

	def check_seq(self, y, y_): 
		min_len = min([len(y), len(y_)]) 
		if sum(np.equal(np.array(y[:min_len], dtype=object), np.array(y_[:min_len]), dtype=object)) == len(y): 
			return 1
		return 0

	def get_token_acc(self):
		a = 0
		for i in range(self.size):
			y = self.ys[i]
			y_ = self.ys_[i]
			a += self.check_token(y, y_)
		return np.float32(a/self.size)

	def get_seq_acc(self): 
		a = 0
		for i in range(self.size):
			y = self.ys[i]
			y_ = self.ys_[i]
			a += self.check_seq(y, y_)
		return np.float32(a/self.size)

	def get_eq_acc(self):
		a = 0
		for i in range(self.size):
			y = self.ys[i]
			y_ = self.ys_[i]
			a += self.check_equation(y, y_)
		return np.float32(a/self.size)

	def check_equation(self, tgt, pred): 
		# e.g., ['3', '3', '9', '3', '6']
		tgt_nums = [t for t in tgt if t.isdigit()]
		# e.g., ['3', '3', '9', '3', '6']
		pred_nums = [p for p in pred if p.isdigit()]
		# eval('123') return 123
		# eval('1 2 3') raise error
		try:
			if tgt_nums == pred_nums and pred[-1].isdigit() and pred[-2] == '==':
					if eval(''.join(pred)):
						return 1
		except:
			return 0
		return 0