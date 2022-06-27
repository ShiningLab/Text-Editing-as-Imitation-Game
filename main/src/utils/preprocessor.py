#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


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

def pun_preprocessor(raw_xs, raw_ys, tokenizer, env, config):
	# convert COMMA, PERIOD and QUESTION to punctuations
	pun_dict = {}
	pun_dict[env.COMMA_TOKEN] = ','
	pun_dict[env.PERIOD_TOKEN] = '.'
	pun_dict[env.QUESTION_TOKEN] = '?'
	xs = []
	ys = None if raw_ys is None else []
	for i in range(len(raw_xs)):
		x = ' '.join(raw_xs[i])
		if env.puns & set(raw_xs[i]):
			for k in pun_dict:
				x = x.replace(k, pun_dict[k])
		vocab_dict_key = 'src_token2idx' if config.eva else 'tgt_token2idx'
		xs.append(x.split())
		if ys is not None:
			y = raw_ys[i].copy()
			y = helper.translate(y, env.vocab_dict[vocab_dict_key])
			ys.append(y)
	xs, x_masks = encode_for_lm(xs, tokenizer, env, config)
	if ys is None:
		# state, state mask
		return xs, x_masks
	else:
		# state, state attention mask, action
		return xs, x_masks, ys

def encode_for_lm(raw_xs, tokenizer, env, config, batch_max_len=False):
		xs, x_masks = [], []
		if batch_max_len:
			max_seq_len = max([len(x) for x in raw_xs])
		else:
			max_seq_len = config.src_seq_len
		for i in range(len(raw_xs)):
			x = [config.BOS_TOKEN] + raw_xs[i] + [config.EOS_TOKEN]
			# padding
			if len(x) < max_seq_len:
				pads = [config.PAD_TOKEN for _ in range(max_seq_len - len(x))]
				x += pads
			x_mask = [0 if token == config.PAD_TOKEN else 1 for token in x]
			x = tokenizer.convert_tokens_to_ids(x)
			xs.append(x)
			x_masks.append(x_mask)
		return xs, x_masks