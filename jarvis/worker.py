# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:31:04 2020

@author: Zhe
"""

import os, time

from .archive import Archive
from .utils import set_seed

class ModelTrainer:
    def __init__(self, save_dir, get_config, print_info, preprocess):
        self.save_dir = save_dir
        
        self.configs = Archive(os.path.join(save_dir, 'configs'))
        self.stats = Archive(os.path.join(save_dir, 'stats'), max_try=20)
        self.ckpts = Archive(os.path.join(save_dir, 'ckpts'), f_name_len=4, max_try=20, pause=1.)
        
        self.get_config = get_config
        self.print_info = print_info
        self.preprocess = preprocess
    
    @staticmethod
    def is_valid(asgmt_config):
        if not isinstance(asgmt_config, dict):
            return False
        if 'train_config' not in asgmt_config:
            return False
        if 'seed' not in asgmt_config['train_config']:
            return False
        return True
    
    def is_completed(self, a_id):
        return self.stats.has_id(a_id) and self.stats.fetch_record(a_id)['completed'] and self.ckpts.has_id(a_id)
    
    def process(self, asgmt_config, run_config):
        self.print_info(asgmt_config)
        
        a_id = self.configs.add(asgmt_config)
        if self.is_completed(a_id):
            if 'ignore_existing' in run_config and run_config['ignore_existing']:
                print('a completed assignment ({}) exists, will be replaced')
            else:
                print('a completed assignment ({}) exists, will skip')
                return
        
        if self.preprocess is not None:
            self.preprocess(asgmt_config, run_config)
        
        print('set seed and start processing')
        set_seed(asgmt_config['train_config']['seed'])
        tic = time.time()
        
        print('prepare datasets')
        
        print('prepare models')
        
        print('define criterion')
        
        print('log initial states and evaluations')
        # epoch_num = 0
        # states = []
        # losses = {'valid': [], 'test': []}
        
        print('prepare optimizer')
        
        print('prepare controller')
        
        print('main loop')
        # while True:
        #     # completed, last_idx = controller(losses['valid'])
        #     if completed or epoch_num%save_period==0:
        #         # save progress
            
        #     if completed:
        #         break
            
        #     if last_idx<epoch_num:
        #         # reset to last_idx
            
        #     tic_epoch = time.time()
        #     epoch_num += 1
        #     print('\nepoch {}'.format(epoch_num))
            
        #     # train
            
        #     # evaluate and log
        #     toc_epoch = time.time()
        #     print('elapsed time for one epoch: {}'.format(time_str(toc_epoch-tic_epoch)))
        
        print('display brief summary')
        
        print('return ID')
        return
