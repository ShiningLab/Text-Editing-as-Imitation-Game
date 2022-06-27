#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'

# helper functions
def save_txt(path: str, line_list:list) -> None:
    with open(path, 'w', encoding='utf-8') as f: 
        for line in line_list: 
            f.write(line + '\n') 
    f.close()

def convert_to_str(seq:list) -> str:
    seq = [str(number) for number in seq]
    return ' '.join(seq) 