#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import pickle
import random
import itertools
from collections import Counter
# public
import numpy as np
import torch
# private
from envs import aor, aes, aec, pun
from src.models import (base_seq2seq_lstm_model, seq2seq_lstm_model
	, base_lstm_model, lstm_model, lm_model, lm_lstm_model
	, lstm0_model, lstm1_model, lstm2_model)
from src.trainers import seq2seq_trainer, lm_trainer
from src.utils import augmentor, datasets, preprocessor, postprocessor


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def save_pickle(path, obj):
	with open(path, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
	with open(path, 'rb') as f:
		obj = pickle.load(f)
	return obj

def get_env(config):
	if config.env == 'aor':
		return aor.GameEnv(metric=config.metric)
	elif config.env == 'aes':
		return aes.GameEnv(metric=config.metric)
	elif config.env == 'aec':
		return aec.GameEnv(metric=config.metric)
	elif config.env == 'pun':
		return pun.GameEnv(metric=config.metric)
	else:
		raise NotImplementedError

def get_model(config):
	if config.model == 'base_seq2seq_lstm':
		return base_seq2seq_lstm_model.ModelGraph(config)
	elif config.model == 'seq2seq_lstm':
		return seq2seq_lstm_model.ModelGraph(config)
	elif config.model == 'base_lstm':
		return base_lstm_model.ModelGraph(config)
	elif config.model == 'lstm':
		return lstm_model.ModelGraph(config)
	elif config.model == 'lm':
		return lm_model.ModelGraph(config)
	elif config.model == 'lm_lstm':
		return lm_lstm_model.ModelGraph(config)
	elif config.model == 'lstm0':
		return lstm0_model.ModelGraph(config)
	elif config.model == 'lstm1':
		return lstm0_model.ModelGraph(config)
	elif config.model == 'lstm2':
		return lstm0_model.ModelGraph(config)
	else:
		raise NotImplementedError

def get_trainer(config):
	if config.model in ['lm', 'lm_lstm']:
		return lm_trainer
	else:
		return seq2seq_trainer

def get_dataset(config):
	if config.env in ['aor', 'aes', 'aec']:
		return datasets.AEDataset
	elif config.env == 'pun':
		return datasets.PunDataset
	else:
		raise NotImplementedError

def get_augmentor(config):
	if config.env in ['aor', 'aes', 'aec']:
		return augmentor.generic_augmentor
	elif config.env == 'pun':
		return augmentor.online_pun_augmentor
	else:
		raise NotImplementedError

def get_preprocessor(config):
	if config.env in ['aor', 'aes', 'aec']:
		return preprocessor.ae_preprocessor
	elif config.env == 'pun':
		return preprocessor.pun_preprocessor
	else:
		raise NotImplementedError

def get_postprocessor(config):
	return postprocessor

def init_parameters(model): 
	for name, parameters in model.named_parameters(): 
		if 'weight' in name: 
			torch.nn.init.normal_(parameters.data, mean=0, std=0.01)
		else:
			torch.nn.init.constant_(parameters.data, 0)

def count_parameters(model): 
	# get total size of trainable parameters 
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_vocab(env, config):
	# vocabulary frequency distribution
	src_counter = Counter()
	tgt_counter = Counter()
	for mode in ['train', 'valid', 'test']:
		for states, actions in zip(env.data_dict[mode]['xs'], env.data_dict[mode]['ys']):
			for state, action in zip(states, actions):
				src_counter.update(state)
				tgt_counter.update(action)
	src_token_list = sorted(src_counter.keys())
	tgt_token_list = sorted(tgt_counter.keys())
	# extend full position for trajectory augmentation
	for i in range(env.max_seq_len+1):
		pos = '<pos_{}>'.format(i)
		if pos not in tgt_token_list:
			tgt_token_list.append(pos)
	# source vocabulary dictionary
	src_token2idx_dict = dict()
	# to pad sequence length
	src_token2idx_dict[config.PAD_TOKEN] = 0
	i = len(src_token2idx_dict)
	for token in src_token_list:
		src_token2idx_dict[token] = i
		i += 1
	# target vocabulary dictionary
	tgt_token2idx_dict = dict()
	# BOS token
	if config.model not in ['base_lstm', 'lstm0', 'lstm1']:
		tgt_token2idx_dict[config.BOS_TOKEN] = 0
	i = len(tgt_token2idx_dict)
	for token in tgt_token_list:
		tgt_token2idx_dict[token] = i
		i += 1
	vocab_dict = {}
	vocab_dict['src_token2idx'] = src_token2idx_dict
	vocab_dict['tgt_token2idx'] = tgt_token2idx_dict
	vocab_dict['src_idx2token'] = {v:k for k, v in src_token2idx_dict.items()}
	vocab_dict['tgt_idx2token'] = {v:k for k, v in tgt_token2idx_dict.items()}
	return vocab_dict

def flatten_list(l):
	return list(itertools.chain(*l))

def unique_lists(xs, ys):
	pairs = []
	for x, y in zip(xs, ys):
		if (x, y) not in pairs:
			pairs.append((x, y))
	pairs.sort(key=lambda item: len(item[0]))
	return map(list, zip(*pairs))

def translate(seq: list, trans_dict: dict) -> list: 
	return [trans_dict[token] for token in seq]

def show_rand_sample(srcs, tars, preds): 
	src, tar, pred = random.choice([(src, tar, pred) for src, tar, pred in zip(srcs, tars, preds)])
	src, tar, pred = ' '.join(src), ' '.join(tar), ' '.join(pred)
	print(' src: {}\n tar: {}\n pred: {}'.format(src, tar, pred))

def save_check_point(step, epoch, model_state_dict, opt_state_dict, log_dict, path, lm=None, lm_path=None):
	# save model, optimizer, and everything required to keep
	checkpoint_to_save = {
		'step': step
		, 'epoch': epoch
		, 'model': model_state_dict()
		, 'optimizer': opt_state_dict()
		, 'log_dict': log_dict}
	torch.save(checkpoint_to_save, path)
	if lm_path:
		lm.save_pretrained(lm_path)
	print('Model saved as {}.'.format(path))

def show_config(config):
	# general information
	general_info = '\n*Configuration*'
	general_info += '\ndevice: {}'.format(config.device)
	general_info += '\nrandom seed: {}'.format(config.seed)
	general_info += '\ngame: {}'.format(config.env)
	general_info += '\nsrc vocab size: {}'.format(config.src_vocab_size)
	general_info += '\ntgt vocab size: {}'.format(config.tgt_vocab_size)
	general_info += '\nmodel: {}'.format(config.model)
	general_info += '\nlanguage model: {}'.format(config.lm)
	general_info += '\ntrainable parameters: {:,.0f}'.format(config.num_parameters)
	general_info += '\nmax trajectory length: {}'.format(config.max_infer_step)
	general_info += '\nmax state sequence length: {}'.format(config.max_seq_len)
	general_info += '\nmax action sequence length: {}'.format(config.tgt_seq_len)
	general_info += '\nif load check point: {}'.format(config.load_ckpt)
	general_info += '\noriginal train size: {}'.format(config.ori_train_size)
	general_info += '\ntrain size: {}'.format(config.train_size)
	general_info += '\nvalid size: {}'.format(config.valid_size)
	general_info += '\ntest size: {}'.format(config.test_size)
	general_info += '\nif sampling: {}'.format(config.sampling)
	general_info += '\nbatch size: {}'.format(config.batch_size)
	general_info += '\nlearning rate: {}'.format(config.learning_rate)
	general_info += '\n'
	print(general_info)

	return general_info

# adapted from https://github.com/ShiningLab/Recurrent-Text-Editing/blob/master/main/src/utils/pipeline.py
def rm_idx(seq, idx):
	return [i for i in seq if i != idx]

def parse_pos(pos):
	return int(''.join([i for i in pos if i.isdigit()]))

def padding(xs, max_len, config):
	x_lens = [min(len(x), max_len) for x in xs]
	padded_xs = torch.zeros([len(xs), max_len]).fill_(config.PAD_IDX).long()
	for i, x in enumerate(xs): 
		x_len = x_lens[i]
		padded_xs[i, :x_len] = x[:x_len]
	if 'lstm' in config.model:
		return padded_xs.to(config.device), torch.Tensor(x_lens)
	else:
		raise NotImplementedError

# without language model
def rec_infer(xs, xs_info, env, model, max_infer_step, config, done=False):
	# recursive inference in valudation and testing
	if max_infer_step == 0:
		return xs, xs_info, False
	else:
		xs, xs_info, done = rec_infer(xs, xs_info, env, model, max_infer_step-1, config, done) 
		if done:
			return xs, xs_info, done
		else:
			if xs_info is not None:
				ys_ = model(xs, xs_info)
			else:
				ys_ = model(xs)
			(xs, xs_info), done = one_step_infer(xs, ys_, env, config)
			return xs, xs_info, done

def action_processor(ys_, env):
	ys_ = torch.argmax(ys_, dim=-1).cpu().detach().numpy().tolist()
	ys_ = [helper.translate(y_, env.vocab_dict['tgt_idx2token']) for y_ in ys_]
	return ys_

def one_step_infer(xs, ys_, env, config):
	# detach from devices
	xs = xs.cpu().detach().numpy()
	# remove padding idx
	xs = [rm_idx(x, config.PAD_IDX) for x in xs] 
	# convert index to vocab
	xs = [translate(x, env.vocab_dict['src_idx2token']) for x in xs]
	ys_ = postprocessor.action_processor(ys_, env)
	# mask completed sequences
	mask = ~(np.array(ys_) == env.DONE).all(axis=-1)
	if not mask.any():
		done = True
	else:
		done = False
		idxes = np.arange(len(xs))[mask]
		for i in idxes:
			xs[i] = env.one_step_infer(xs[i], ys_[i])
	xs = [torch.Tensor(translate(x, env.vocab_dict['src_token2idx'])) for x in xs]
	return padding(xs, config.max_seq_len, config), done

# for language model
def lm_rec_infer(raw_xs, raw_x_masks, xs, x_masks, max_infer_step, model, env, preprocessor, tokenizer, config, done=False):
	# recursive inference in valudation and testing
	if max_infer_step == 0:
		return raw_xs, raw_x_masks, xs, x_masks, False
	else:
		raw_xs, raw_x_masks, xs, x_masks, done = lm_rec_infer(
			raw_xs, raw_x_masks, xs, x_masks, max_infer_step-1
			, model, env, preprocessor, tokenizer, config, done) 
		if done:
			return raw_xs, raw_x_masks, xs, x_masks, done
		else:
			ys_ = model(xs, x_masks)
			return lm_one_step_infer(
				raw_xs, raw_x_masks, x_masks, ys_
				, env, preprocessor, tokenizer, config)

def lm_one_step_infer(raw_xs, raw_x_masks, x_masks, ys_, env, preprocessor, tokenizer, config): 
	ys_ = postprocessor.action_processor(ys_, env)
	# mask completed sequences
	mask = ~(np.array(ys_) == env.DONE).all(axis=-1)
	if not mask.any():
		done = True
	else:
		done = False
		idxes = np.arange(len(raw_xs))[mask]
		if config.env == 'pun':
			for i in idxes:
				if len(raw_xs[i]) < config.max_seq_len:
					raw_xs[i], raw_x_masks[i] = env.one_step_infer(raw_xs[i], raw_x_masks[i], ys_[i])
		else:
			raise NotImplementedError
	xs, x_masks = preprocessor(raw_xs, None, tokenizer, env, config)
	xs, x_masks = (torch.LongTensor(_).to(config.device) for _ in (xs, x_masks))
	return raw_xs, raw_x_masks, xs, x_masks, done