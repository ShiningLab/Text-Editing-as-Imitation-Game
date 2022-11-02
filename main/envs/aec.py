#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# built-in
import os
# private
from envs.distance import Editops
from envs.base import TextEditEnv
from envs.utils import load_txt_to_list, parse_pos


class GameEnv(TextEditEnv):
    """
    The game environment class for Arithmetic Equation Correction (AEC)
    """
    def __init__(self, metric='levenshtein'):
        super(GameEnv, self).__init__()
        self.env = 'aec'
        # depend on action definition
        # 3 for [operation, position, token]
        self.tgt_seq_len = 3
        self.ops = set(['<replace>', '<delete>', '<insert>'])
        self.operators = set(['+', '-', '*', '/', '=='])
        self.traj_generator = Editops(metric=metric)

    def get_data(self, N, L, D):
        RAW_AE = os.path.join(self.RAW_DATA_PATH, 'arithmetic_equation')
        RAW_AOR = os.path.join(RAW_AE, self.env, '{}N'.format(N), '{}L'.format(L), '{}D'.format(D))
        raw_train_xs = load_txt_to_list(os.path.join(RAW_AOR, 'train_x.txt'))
        raw_train_ys = load_txt_to_list(os.path.join(RAW_AOR, 'train_y.txt'))
        raw_valid_xs = load_txt_to_list(os.path.join(RAW_AOR, 'val_x.txt'))
        raw_valid_ys = load_txt_to_list(os.path.join(RAW_AOR, 'val_y.txt'))
        raw_test_xs = load_txt_to_list(os.path.join(RAW_AOR, 'test_x.txt'))
        raw_test_ys = load_txt_to_list(os.path.join(RAW_AOR, 'test_y.txt'))
        # save data size
        self.data_size_dict = {}
        self.data_size_dict['train'] = len(raw_train_xs)
        self.data_size_dict['valid'] = len(raw_valid_xs)
        self.data_size_dict['test'] = len(raw_test_xs)
        return (raw_train_xs, raw_train_ys, raw_valid_xs, raw_valid_ys, raw_test_xs, raw_test_ys)
    
    def make(self, N=10, L=5, D=10000):
        self.numbers = set([str(n) for n in range(2, N+2)])
        data = self.get_data(N, L, D)
        self.data_dict = {}
        for mode, (xs, ys) in zip(['train', 'valid', 'test'], [(data[0], data[1]), (data[2], data[3]), (data[4], data[5])]):
            states, actions = map(list, zip(*[self.get_trajectories(x, y) for x, y in zip(xs, ys)]))
            self.data_dict[mode] = {}
            self.data_dict[mode]['xs'] = states
            self.data_dict[mode]['ys'] = actions
        self.max_seq_len = max([len(x) for xs in self.data_dict['train']['xs'] for x in xs])
        self.max_infer_step = max([len(xs) for xs in self.data_dict['train']['xs']])

    def get_trajectories(self, x, y, do_copy=True):
        if isinstance(x, str):
            # white space tokenization
            x, y = x.split(), y.split()
        # to avoid mutable issue
        if do_copy:
            x, y = x.copy(), y.copy()
        # initialize placeholder
        xs = [x.copy()]
        ys_ = []
        editops = self.traj_generator.editops(x, y)
        c = 0
        # operation, index i, index j
        for op, i, j in editops:
            i += c
            if op == 'replace':
                y_ = ['<replace>', '<pos_{}>'.format(i), y[j]]
                x[i] = y[j]
            elif op == 'delete':
                y_ = ['<delete>', '<pos_{}>'.format(i), '<pos_{}>'.format(i)]
                del x[i]
                c -= 1
            elif op == 'insert':
                y_ = ['<insert>', '<pos_{}>'.format(i), y[j]]
                x.insert(i, y[j])
                c += 1
            else:
                raise NotImplementedError
            xs.append(x.copy())
            ys_.append(y_)
        # DONE is the special mark for the final action
        ys_.append([self.DONE] * self.tgt_seq_len)
        return xs, ys_

    def one_step_infer(self, state, action, do_copy=True):
        # to avoid mutable issue
        if do_copy:
            state = state.copy()
        # check if final action
        if action.count(self.DONE) == len(action):
            return state
        # operation, position, target token
        # for delete operation, both position and target token are the same position
        op, pos, tk = action
        if op in self.ops and pos.startswith('<pos_'):
            pos = parse_pos(pos)
            if op == '<replace>' and pos in range(len(state)) and tk in self.numbers | self.operators:
                state[pos] = tk
            elif op == '<delete>' and tk.startswith('<pos_'):
                tk = parse_pos(tk)
                if pos in range(len(state)) and pos == tk:
                    del state[pos]
            elif op == '<insert>' and pos in range(len(state)+1) and tk in self.numbers | self.operators:
                state.insert(pos, tk)
            else:
                pass
        return state