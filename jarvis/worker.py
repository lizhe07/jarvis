# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:31:04 2020

@author: Zhe
"""

from .archive import Archive
from .utils import set_seed

class ModelTrainer:
    def __init__(self, save_dir, get_config):
        self.save_dir = save_dir
        
        self.configs = Archive(save_dir)
        self.stats = Archive(save_dir, max_try=20)
        self.ckpts = Archive(save_dir, f_name_len=4, max_try=20, pause=1.)
        
        self.get_config = get_config
    
    # def hypersearch(self, search_spec):
    #     print('sample argument strings')
    #     # arg_strs = self.sample(search_spec)
        
    #     print('get assignment configuration')
        
    
    def main(self, assign_config, run_config):
        print('display assignment info')
        # self.print_info(assign_config)
        
        print('get assignment id and check existing')
        # a_id = self.configs.add(assign_config)
        # if self.stats.has_id(a_id) and self.stats.fetch_record(a_id)['completed'] and self.ckpts.has_id(a_id):
        #     if 'ignore_existing' in run_config and run_config['ignore_existing']:
        #         print('a completed assignment ({}) exists, will be replaced')
        #     else:
        #         print('a completed assignment ({}) exists, will skip')
        #         return
        
        print('prepare certain data')
        # self.preprocess(assign_config, run_config)
        
        print('set seed and start processing')
        # set_seed(assign_config['train_config']['seed'])
        # tic = time.time()
        
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
