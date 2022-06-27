#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
# public
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


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
		if config.env in ['aor', 'aes', 'aec']:
			self.token_acc = self.get_token_acc()
			self.seq_acc = self.get_seq_acc()
			self.eq_acc = self.get_eq_acc()
			self.key_metric = self.token_acc if config.env == 'aes' else self.eq_acc
			self.keys = ['token_acc', 'seq_acc', 'eq_acc']
			self.values = [self.token_acc, self.seq_acc, self.eq_acc]
			self.msg = ' Token Acc:{:.4f} Seq Acc:{:.4f} Eq Acc:{:.4f}'.format(self.token_acc, self.seq_acc, self.eq_acc)
		elif config.env == 'pun':
			labels = ['COMMA', 'PERIOD', 'QUESTION']
			self.ys = pun_postprocess(self.ys, labels)
			self.ys_ = pun_postprocess(self.ys_, labels)
			# precision
			self.comma_precision, self.period_precision, self.question_precision = self.get_precision(labels)
			# recall
			self.comma_recall, self.period_recall, self.question_recall = self.get_recall(labels)
			# f1
			self.comma_f1, self.period_f1, self.question_f1 = self.get_f1(labels)
			# macro f1
			self.macro_f1 = np.mean([self.comma_f1, self.period_f1, self.question_f1])
			# micro f1
			self.micro_precision = precision_score(self.ys, self.ys_, average='micro', zero_division=0., labels=labels)
			self.micro_recall = recall_score(self.ys, self.ys_, average='micro', zero_division=0., labels=labels)
			if self.micro_precision * self.micro_recall:
				self.micro_f1 = 2 * (self.micro_precision * self.micro_recall) / (self.micro_precision + self.micro_recall)
			else:
				self.micro_f1 = 0.
			self.key_metric = self.micro_f1
			self.keys = ['comma_precision', 'comma_recall', 'comma_f1']
			self.keys += ['period_precision', 'period_recall', 'period_f1']
			self.keys += ['question_precision', 'question_recall', 'question_f1']
			self.keys += ['macro_f1', 'micro_precision', 'micro_recall', 'micro_f1']
			self.values = [self.comma_precision, self.comma_recall, self.comma_f1]
			self.values += [self.period_precision, self.period_recall, self.period_f1]
			self.values += [self.question_precision, self.question_recall, self.question_f1]
			self.values += [self.macro_f1, self.micro_precision, self.micro_recall, self.micro_f1]
			comma_msg = 'COMMA P:{:.4f} R:{:.4f} F:{:.4f}'.format(self.comma_precision, self.comma_recall, self.comma_f1)
			period_msg = 'PERIOD P:{:.4f} R:{:.4f} F:{:.4f}'.format(self.period_precision, self.period_recall, self.period_f1)
			question_msg = 'QUESTION P:{:.4f} R:{:.4f} F:{:.4f}'.format(self.question_precision, self.question_recall, self.question_f1)
			avg_msg = 'Macro F:{:.4f} Micro P:{:.4f} R:{:.4f} F:{:.4f}'.format(self.macro_f1, self.micro_precision, self.micro_recall, self.micro_f1)
			self.msg = '\n' + comma_msg + '\n' + period_msg + '\n' + question_msg + '\n' + avg_msg
		else:
			raise NotImplementedError

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

	def get_precision(self, labels):
		return precision_score(self.ys, self.ys_, average=None, zero_division=0., labels=labels)

	def get_recall(self, labels):
		return recall_score(self.ys, self.ys_, average=None, zero_division=0., labels=labels)

	def get_f1(self, labels):
		return f1_score(self.ys, self.ys_, average=None, zero_division=0., labels=labels)

def pun_postprocess(ys, labels):
    ys = [l for y in ys for l in y]
    for i in range(len(ys)):
        if ys[i] not in labels:
            ys[i] = 'O'
        else:
            ys[i-1] = None
    ys = [_ for _ in ys if _]
    return ys