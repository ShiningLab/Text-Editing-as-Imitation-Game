#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# dependency
# built-in
import os
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# On CUDA 10.2 or later, set environment variable (note the leading colon symbol) 
# CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:2
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
from datetime import datetime
# public
import torch
from torch.utils import data as torch_data
import numpy as np
rng = np.random.default_rng()
from tqdm import tqdm
# private
from src.utils import helper, eva
from src.trainers.base_trainer import Base_Trainer


class Trainer(Base_Trainer):
	"""docstring for Trainer"""
	def __init__(self, model, env, config, **kwargs):
		super().__init__(model, env, config)
		self.update_config(**kwargs)
		self.initialize()
		self.setup_dataloader()

	def update_config(self, **kwargs):
		# configuration for lstm
		self.config.batch_size = 256
		self.config.learning_rate = 1e-3
		# update configuration accordingly
		for k,v in kwargs.items():
			setattr(self.config, k, v)

	def initialize(self):
		# data augmentation
		self.augmentor = helper.get_augmentor(self.config) if self.config.aug_trajectory else None
		# dataset class
		self.Dataset = helper.get_dataset(self.config)
		# data process
		self.pre_process = helper.get_preprocessor(self.config)
		self.post_process = helper.get_postprocessor(self.config)
		# optimization
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
		self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
			optimizer=self.optimizer
			, T_0=32
			)
		# loss function
		if self.config.adapt_loss:
			self.criterion = torch.nn.NLLLoss(reduction='none')
			self.loss_w = torch.Tensor([1.] * self.config.tgt_seq_len).to(self.config.device)
		else:
			self.criterion = torch.nn.NLLLoss()
		# if load check point
		if self.config.load_ckpt:
			self.load_ckpt()

	def collate_fn(self, data):
		# a customized collate function used in the data loader 
		# without language model
		data.sort(key=len, reverse=True)
		# base trajectories of batch size
		raw_xs, raw_ys = zip(*data)
		# initial state to goal state for inference
		if self.config.eva:
			raw_xs, raw_ys = zip(*[(x[0], x[-1]) for x in raw_xs])
		else:
			# batch size to trajectories size
			raw_xs, raw_ys = helper.flatten_list(raw_xs), helper.flatten_list(raw_ys)
			raw_xs, raw_ys = helper.unique_lists(raw_xs, raw_ys)
			if self.config.sampling:
				raw_xs_ys = [(x, y) for x, y in zip(raw_xs, raw_ys)]
				rng.shuffle(raw_xs_ys)
				raw_xs, raw_ys = zip(*raw_xs_ys[:self.config.batch_size])
		# prepare data
		xs, x_lens, ys = self.pre_process(raw_xs, raw_ys, self.env, self.config)
		return (raw_xs, raw_ys), (xs, x_lens, ys)

	def setup_dataloader(self):
		train_dataset = self.Dataset(self.env, self.augmentor, 'train', self.config)
		train_dataloader = torch_data.DataLoader(
			train_dataset
			, batch_size=self.config.batch_size
			, collate_fn=self.collate_fn
			, shuffle=self.config.shuffle
			, num_workers=self.config.num_workers
			, pin_memory=self.config.pin_memory
			, drop_last=self.config.drop_last
			)
		self.dataloader_dict['train'] = train_dataloader
		for mode in ['valid', 'test']:
			dataset = self.Dataset(self.env, None, mode, self.config)
			dataloader = torch_data.DataLoader(
				dataset
				, batch_size=self.config.batch_size
				, collate_fn=self.collate_fn
				, shuffle=False
				, num_workers=self.config.num_workers
				, pin_memory=self.config.pin_memory
				, drop_last=False
				)
			self.dataloader_dict[mode] = dataloader
		# update config
		self.config.train_size = len(train_dataset)

	def run_epoch(self, mode):
		epoch_loss, epoch_steps, epoch_xs, epoch_ys, epoch_ys_ = 0., 0, [], [], []
		dataloader = self.dataloader_dict[mode]
		dataloader = tqdm(dataloader)
		for (raw_xs, raw_ys), (xs, x_lens, ys) in dataloader:
			xs, ys = (torch.LongTensor(_).to(self.config.device) for _ in (xs, ys))
			ys_ = self.model(xs, torch.Tensor(x_lens), None if self.config.eva else ys)
			# release cache
			del xs
			torch.cuda.empty_cache()
			loss = self.criterion(ys_.reshape(-1, self.config.tgt_vocab_size), ys.reshape(-1))
			if self.config.adapt_loss:
				loss = (loss.reshape(-1,  self.config.tgt_seq_len) * self.loss_w).mean()
			if mode == 'train':
				loss.backward()
				if self.config.clipping_threshold:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clipping_threshold)
				self.optimizer.step()
				self.optimizer.zero_grad()
				self.step += 1
			loss = loss.item()
			dataloader.set_description('{} Loss:{:.4f} LR:{:.6f}'.format(
				mode.capitalize(), loss, self.scheduler.get_last_lr()[0]))
			# post processing
			ys_ = self.post_process.action_processor(ys_, self.env)
			# record
			epoch_loss += loss
			epoch_steps += 1
			epoch_xs += raw_xs
			epoch_ys += raw_ys
			epoch_ys_ += ys_
			# break
		return epoch_loss / epoch_steps, epoch_xs, epoch_ys, epoch_ys_

	def train(self):
		self.log_dict['config'] = helper.show_config(self.config)
		self.log_dict['start_time'] = datetime.now()
		while True:
			self.model.train()
			loss, xs, ys, ys_ = self.run_epoch('train')
			eva_metrics = eva.ActionEvaluater(ys, ys_)
			for k, v in zip(['loss', 'token_acc', 'seq_acc'], [loss, eva_metrics.token_acc, eva_metrics.seq_acc]):
				self.log_dict['train'][k].append(v)
			print('Train Epoch {} Total Step {} Loss:{:.4f} Token Acc:{:.4f} Seq Acc:{:.4f}'.format(
				self.epoch, self.step, loss, eva_metrics.token_acc, eva_metrics.seq_acc))
			# random sample to show
			helper.show_rand_sample(xs, ys, ys_)
			# update adaptive loss weight
			if self.config.adapt_loss:
				self.loss_w = torch.Tensor(np.sum(np.array(ys) == np.array(ys_), axis=0)).to(self.config.device)
				self.loss_w = 1 - self.loss_w/len(ys)
				self.loss_w = self.loss_w / sum(self.loss_w)
				print('Adaptive loss weight updated:', self.loss_w)
			# valid & test
			self.valid_and_test()
			# evaluation
			self.eva()
			# update
			self.scheduler.step()
			self.epoch += 1
			# early stopping
			if self.valid_epoch >= self.config.valid_patience:
				self.log_dict['end_time'] = datetime.now()
				helper.save_pickle(self.config.LOG_SAVE_POINT, self.log_dict)
				break
			# break
		
	def valid_and_test(self):
		self.model.eval()
		with torch.no_grad():
			for mode in ['valid', 'test']:
				loss, xs, ys, ys_ = self.run_epoch(mode)
				eva_metrics = eva.ActionEvaluater(ys, ys_)
				for k, v in zip(['loss', 'token_acc', 'seq_acc'], [loss, eva_metrics.token_acc, eva_metrics.seq_acc]):
					self.log_dict[mode][k].append(v)
				for k, v in zip(['xs', 'ys', 'ys_'], [xs, ys, ys_]):
					self.log_dict[mode][k] = v
				print('{} Epoch {} Total Step {} Loss:{:.4f} Token Acc:{:.4f} Seq Acc:{:.4f}'.format(
					mode.capitalize(), self.epoch, self.step, loss, eva_metrics.token_acc, eva_metrics.seq_acc))
				# random sample to show
				helper.show_rand_sample(xs, ys, ys_)

	def eva(self):
		self.model.eval()
		self.config.eva = True
		metric_dict = dict()
		for mode in ['valid', 'test']:
			dataloader = self.dataloader_dict[mode]
			dataloader = tqdm(dataloader)
			with torch.no_grad():
				all_xs, all_ys, all_ys_ = [], [], []
				for raw_data, (xs, x_lens, _) in dataloader:
					xs = torch.LongTensor(xs).to(self.config.device)
					x_lens = torch.Tensor(x_lens)
					ys_ = helper.rec_infer(xs, x_lens, self.env, self.model, self.config.max_infer_step, self.config)[0]
					# post processing
					ys_ = self.post_process.ae_processor(ys_, self.env, self.config)
					raw_xs, raw_ys = raw_data
					all_xs += raw_xs
					all_ys += raw_ys
					all_ys_ += ys_
					# break
			eva_metrics = eva.Evaluater(all_ys, all_ys_, self.config)
			metric_dict[mode] = eva_metrics
			for k, v in zip(eva_metrics.keys, eva_metrics.values):
				if k not in self.log_dict['eva'][mode]:
					self.log_dict['eva'][mode][k] = []
				self.log_dict['eva'][mode][k].append(v)
			msg = 'Evaluation {} Epoch {} Total Step {}'.format(mode.capitalize(), self.epoch, self.step)
			msg += eva_metrics.msg
			print(msg)
			# random sample to show
			helper.show_rand_sample(all_xs, all_ys, all_ys_)
		# early stopping base
		if metric_dict['valid'].key_metric >= self.log_dict['eva']['best_valid']['valid_key_metric']:
			for k1, k2 in zip(['valid_key_metric', 'test_key_metric'], ['valid', 'test']):
				self.log_dict['eva']['best_valid'][k1] = metric_dict[k2].key_metric
			self.valid_epoch = 0
			for k, v in zip(metric_dict['valid'].keys, metric_dict['valid'].values):
				self.log_dict['eva']['best_valid'][k] = v
			for k, v in zip(['epoch', 'step', 'xs', 'ys', 'ys_'], [self.epoch, self.step, all_xs, all_ys, all_ys_]):
				self.log_dict['eva']['best_valid'][k] = v
			# save model
			helper.save_check_point(
				self.step, self.epoch, self.model.state_dict, self.optimizer.state_dict, self.log_dict, self.config.CKPT_SAVE_POINT)
			# save log
			self.log_dict['end_time'] = datetime.now()
			helper.save_pickle(self.config.LOG_SAVE_POINT, self.log_dict)
		msg = 'Early Stopping Record'
		for s, k in zip(['Epoch {}', 'Total Step {}', 'Valid Key Metric:{:.4f}', 'Test Key Metric:{:.4f}']
			, ['epoch', 'step', 'valid_key_metric', 'test_key_metric']):
			msg += ' ' + s.format(self.log_dict['eva']['best_valid'][k])
		print(msg + '\n')
		self.valid_epoch += 1
		self.config.eva = False