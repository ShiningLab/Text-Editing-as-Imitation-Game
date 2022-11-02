#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# public
import torch
# private
from src.utils import helper


def ae_preprocessor(raw_xs, raw_ys, env, config, batch_max_len=False):
	xs, x_lens, ys = [], [], []
	if batch_max_len:
		max_seq_len = max([len(x) for x in raw_xs])
	else:
		max_seq_len = config.max_seq_len
	for i in range(len(raw_xs)):
		x, y = raw_xs[i].copy(), raw_ys[i].copy()
		x_len = len(x)
		if len(x) < max_seq_len:
			x += [config.PAD_TOKEN for _ in range(max_seq_len - len(x))]
		# token to index
		x = helper.translate(x, env.vocab_dict['src_token2idx'])
		vocab_dict_key = 'src_token2idx' if config.eva else 'tgt_token2idx'
		y = helper.translate(y, env.vocab_dict[vocab_dict_key])
		xs.append(x)
		x_lens.append(x_len)
		ys.append(y)
	return xs, x_lens, ys