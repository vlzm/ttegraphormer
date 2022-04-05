import sys
import os
sys.path.insert(1, '/home/jovyan/graphormer_v2/')

import torch
import numpy as np
import torch_geometric.datasets
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
import pyximport
from torch_geometric.data import InMemoryDataset, download_url
import pandas as pd
from sklearn import preprocessing

pyximport.install(setup_args={'include_dirs': np.get_include()})
import os.path as osp
from torch_geometric.data import Data
import time
import multiprocessing
from torch_geometric.utils import add_self_loops, negative_sampling
from graphormer.data.wrapper import preprocess_item
from data_class import geo_Omsk
from graphormer.data.collator import collator_graphormer

# def handler():
#     p = multiprocessing.Pool(2)
#     r=p.map(f, data)
#     return r

def update_data_split(raw_data, update_dir, split, collat_flag):
    
    if not os.path.exists(update_dir):
        os.makedirs(update_dir)
    
    graph_count = len(raw_data)
    upd_list = list()
    cpu_count = 256
    p = multiprocessing.Pool(cpu_count)
    for i in range(cpu_count - 1, 500, cpu_count):
        data = list()
        flag = False
        for j in range(i - cpu_count + 1, i +1):
            if j%1000 == 0:
                print(j)
                flag = True
                name = j
            data.append(raw_data[j])
        r = p.map(preprocess_item, data)
        for k in range(0,len(r)):
            upd_list.append(r[k])
            
        # if collat_flag == True:
        #     upd_list = [collator_graphormer([x]) for x in upd_list]
            
        if flag == True:
            torch.save(upd_list, osp.join(update_dir, f'{split}_{name}.pt'))
            upd_list = list()
    
def update_data(raw_dir, update_dir):
    for split in ['train', 'test', 'val']:
        collat_flag = False
        raw_data = geo_Omsk(raw_dir, split = split)
        update_data_split(raw_data, update_dir, split, collat_flag = False)

def cli_main():
    start = time.time()
    root = osp.join('dataset', 'omsk')
    raw_dir = osp.join(root, 'processed', 'data_omsk_1')
    update_dir = osp.join(root, 'processed', 'data_omsk_1_upd')
    update_data(raw_dir, update_dir)
    end = time.time()
    print('total time', end - start)

if __name__ == '__main__':
    cli_main()