#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# dependency
# public 
import torch
# private
from src.utils import helper


def action_processor(ys_, env):
	ys_ = torch.argmax(ys_, dim=-1).cpu().detach().numpy().tolist()
	ys_ = [helper.translate(y_, env.vocab_dict['tgt_idx2token']) for y_ in ys_]
	return ys_

def ae_processor(ys_, env, config):
	ys_ = ys_.cpu().detach().numpy().tolist()
	ys_ = [helper.rm_idx(y_, config.PAD_IDX) for y_ in ys_]
	ys_ = [helper.translate(y_, env.vocab_dict['src_idx2token']) for y_ in ys_]
	return ys_