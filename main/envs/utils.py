#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'author'
__email__ = 'email'


# dependency


def load_txt_to_list(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f: 
        return f.read().splitlines()

def parse_pos(pos):
    return int(''.join([i for i in pos if i.isdigit()]))