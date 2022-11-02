#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# dependency
# built-in
import copy
import numpy as np
# public
from tqdm import tqdm
# private
from envs.utils import parse_pos


def update_position(offset, actions):
	"""
	To update the following action accordingly.
	"""
	actions = copy.deepcopy(actions)
	for action in actions:
		for i, item in enumerate(action):
			if item.startswith('<pos'):
				pos = parse_pos(item)
				action[i] = '<pos_{}>'.format(pos - offset)
	return actions

def augmentation(states, state, goal, actions, env, remove_first_action=False, remove_last_action=True):
	"""
	The function of trajectory augmentation (TA) algorithm.
	"""
	# initialization
	actions = actions.copy()
	if remove_first_action:
		next_state = env.one_step_infer(state, actions.pop(0))
		actions = update_position(len(next_state) - len(state), actions)
	if remove_last_action:
		actions.pop()
	next_state = env.one_step_infer(state, actions.pop(0))
	if next_state != goal:
		states.append(next_state)
	if actions:
		# recursion 1 - if conduct current action
		augmentation(states, next_state, goal, actions.copy(), env, False, False)
		# recursion 2 - if not conduct current action
		offset = len(next_state) - len(state)
		# position update
		actions = update_position(offset, actions)
		augmentation(states, state, goal, actions.copy(), env, False, False)

def generic_augmentor(xs, ys, env):
	total_init_states, total_goal_states = [], []
	for x, y in tqdm(zip(xs, ys), total=len(xs)):
		if len(y) > 2:
			# initial states
			init_state = x[0]
			goal_state = x[-1]
			init_states = []
			augmentation(init_states, init_state, goal_state, y, env)        
			total_init_states += init_states
			# goal states
			total_goal_states += [goal_state] * len(init_states)
	if total_init_states:
		states, actions = map(list, zip(*[env.get_trajectories(x, y) 
										  for x, y in zip(total_init_states, total_goal_states)]))
		return states, actions
	else:
		return [], []