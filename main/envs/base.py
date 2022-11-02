#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# dependency
# build-in
import os
import random
# public
import numpy as np


class TextEditEnv():
    """
    The general game environment class for text editing
    """
    def __init__(self):
        super(TextEditEnv, self).__init__()
        self.init_path()
        self.init_token()
    
    def init_path(self):
        self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.PARENT_PATH = os.path.split(self.CURR_PATH)[0]
        self.RESOURCE_PATH = os.path.join(self.PARENT_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        self.RAW_DATA_PATH = os.path.join(self.DATA_PATH, 'raw_data')
    
    def init_token(self):
        self.DONE = '<done>'
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)