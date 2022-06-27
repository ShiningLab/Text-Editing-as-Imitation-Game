#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


import numpy as np


class Editops:
    def __init__(self, metric):
        '''
        metric: the edit distance algorithm, shoule be one of lcs/levenshtein/damerau-levenshtein.
        '''
        metric = metric.lower()
        if metric != 'lcs' and metric != 'levenshtein' and not metric.startswith('damerau'):
            raise NotImplementedError('the parameter metric shoule be one of lcs/levenshtein/damerau-levenshtein.')
        self.metric = metric
        print('Constructing Editops with Metric %s ...' % metric)
        self.is_damerau = self.metric.startswith('damerau')
    
    def __levenshtein_distance_matrix(self, string1, string2):
        is_damerau = self.is_damerau
        n1 = len(string1)
        n2 = len(string2)
        d = np.zeros((n1 + 1, n2 + 1), dtype=int)
        for i in range(n1 + 1):
            d[i, 0] = i
        for j in range(n2 + 1):
            d[0, j] = j
        for i in range(n1):
            for j in range(n2):
                if string1[i] == string2[j]:
                    cost = 0
                else:
                    cost = 2 if self.metric == 'lcs' else 1
                d[i+1, j+1] = min(d[i, j+1] + 1, # insert
                                d[i+1, j] + 1, # delete
                                d[i, j] + cost) # replace
                if is_damerau:
                    if i > 0 and j > 0 and string1[i] == string2[j-1] and string1[i-1] == string2[j]:
                        d[i+1, j+1] = min(d[i+1, j+1], d[i-1, j-1] + cost) # transpose
        return d
    
    def _get_ops_lcs(self, string1, string2, dist_matrix):
        i, j = dist_matrix.shape
        i -= 1
        j -= 1
        ops = list()
        while i != -1 and j != -1 and i+j > 0:
            index = np.argmin([dist_matrix[i-1, j-1], dist_matrix[i, j-1], dist_matrix[i-1, j]])
            if index == 0:
                if dist_matrix[i, j] <= dist_matrix[i-1, j-1]:
                    i -= 1
                    j -= 1
                    continue
                index = np.argmin([dist_matrix[i, j-1], dist_matrix[i-1, j]]) + 1
            if index == 1:
                ops.insert(0, ('insert', i, j - 1))
                j -= 1
            elif index == 2:
                ops.insert(0, ('delete', i - 1, i - 1))
                i -= 1
        return ops
    
    def _get_ops_lev(self, string1, string2, dist_matrix):
        is_damerau = self.is_damerau
        i, j = dist_matrix.shape
        i -= 1
        j -= 1
        ops = list()
        while i != -1 and j != -1 and i+j > 0:
            if is_damerau:
                if i > 1 and j > 1 and string1[i-1] == string2[j-2] and string1[i-2] == string2[j-1]:
                    if dist_matrix[i-2, j-2] < dist_matrix[i, j]:
                        ops.insert(0, ('transpose', i - 1, i - 2))
                        i -= 2
                        j -= 2
                        continue
            index = np.argmin([dist_matrix[i-1, j-1], dist_matrix[i, j-1], dist_matrix[i-1, j]])
            if index == 0:
                if dist_matrix[i, j] > dist_matrix[i-1, j-1]:
                    ops.insert(0, ('replace', i - 1, j - 1))
                i -= 1
                j -= 1
            if index == 1:
                ops.insert(0, ('insert', i, j - 1))
                j -= 1
            elif index == 2:
                ops.insert(0, ('delete', i - 1, i - 1))
                i -= 1
        return ops

    def execute_ops(self, ops, string1, string2):
        string = list(string1)
        shift = 0
        for op in ops:
            i, j = op[1], op[2]
            if op[0] == 'delete':
                del string[i + shift]
                shift -= 1
            elif op[0] == 'insert':
                string.insert(i + shift, string2[j])
                shift += 1
            elif op[0] == 'replace':
                string[i + shift] = string2[j]
            elif op[0] == 'transpose':
                string[i + shift], string[j + shift] = string[j + shift], string[i + shift]
        return ' '.join(string)
    
    def distance(self, string1, string2):
        n1 = len(string1)
        n2 = len(string2)
        return self.__levenshtein_distance_matrix(string1, string2)[n1, n2]
    
    def editops(self, s1, s2):
        dist_matrix = self.__levenshtein_distance_matrix(s1, s2)
        if self.metric == 'lcs':
            ops = self._get_ops_lcs(s1, s2, dist_matrix)
        else:
            ops = self._get_ops_lev(s1, s2, dist_matrix)
        return ops