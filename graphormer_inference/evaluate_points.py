import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from os import path
# sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import logging
from data_class import geo_Omsk, single_geo_Omsk, GraphormerPYGDataset_predict, single_geo_Abakan
import os.path as osp
from torch_geometric.data import Dataset
from functools import lru_cache
import torch_geometric.datasets
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
import pyximport
from torch_geometric.data import InMemoryDataset, download_url
import pandas as pd
from sklearn import preprocessing
pyximport.install(setup_args={'include_dirs': np.get_include()})
from torch_geometric.data import Data
import time
from torch_geometric.utils import add_self_loops, negative_sampling
import copy
from fairseq.data import (
    NestedDictionaryDataset,
    NumSamplesDataset,
)
import json
import pathlib
link = pathlib.Path().resolve()
link = str(link).split('TransTTE')[0]
GLOBAL_ROOT = link + 'TransTTE'

sys.path.insert(2, GLOBAL_ROOT + '/graphormer_repo/graphormer')
from data.wrapper import preprocess_item

from pretrain import load_pretrained_model
from data.pyg_datasets.pyg_dataset import GraphormerPYGDataset
from data.dataset import (
    BatchedDataDataset,
    TargetDataset,
    GraphormerDataset)

def eval(args, use_pretrained, checkpoint_path=None, logger=None, data_name = None, predict_dataset = None):
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    seed = 71
    
    GPYG = GraphormerPYGDataset_predict(predict_dataset, seed, None, predict_dataset, data_name)
    batched_data = BatchedDataDataset(GPYG)
    data_sizes = np.array([128] * len(batched_data))
    dataset_total = NestedDictionaryDataset(
            {
                "nsamples": NumSamplesDataset(),
                "net_input": {"batched_data": batched_data},
                "target": batched_data,
            },
        sizes=data_sizes,
        )
    ###
    
    ### initialize task
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    batch_iterator = task.get_batch_iterator(
        dataset=dataset_total
    )
    itr = batch_iterator.next_epoch_itr(shuffle=False, set_dataset_epoch=False)
    progress = progress_bar.progress_bar(itr)
    ###
    
    ### load checkpoint
    model_state = torch.load(checkpoint_path)["model"]
    model.load_state_dict(model_state, strict=True, model_cfg=cfg.model)
    model.to(torch.cuda.current_device())
    del model_state
    ###
    
    ### prediction
    y_pred = []
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample)
            y = model(**sample["net_input"])[:, 0, :].reshape(-1)
            y_pred.extend(y.detach().cpu())
            torch.cuda.empty_cache()
    ###
    
    # save predictions
    y_pred = torch.Tensor(y_pred)    
    return y_pred

 


def predict_time(dataset_name, predict_dataset):
    
    parser_dict = dict()
    parser_dict['num-atoms'] = str(6656)
    parser_dict['dataset_name'] = dataset_name
    train_parser = options.get_training_parser()
    train_parser.add_argument(
            "--split",
            type=str,
        )
    train_parser.add_argument(
            "--metric",
            type=str,
        )
    train_parser.add_argument(
            "--dataset_name",
            type=str,
        )
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            '--user-dir' , GLOBAL_ROOT + '/graphormer_repo/graphormer',
            '--num-workers' , '10', 
            '--ddp-backend' , 'legacy_ddp', 
            '--dataset_name' , parser_dict['dataset_name'], 
            '--dataset-source' , 'pyg', 
            '--num-atoms' , parser_dict['num-atoms'], 
            '--task' , 'graph_prediction', 
            '--criterion' , 'l1_loss', 
            '--arch' , 'graphormer_slim',
            '--num-classes' , '1', 
            '--batch-size' , '1', 
            '--save-dir' ,  GLOBAL_ROOT + '/graphormer_repo/examples/georides/omsk/ckpts/',
            '--split' , 'valid', 
            '--metric' , 'rmse',
            '--mode', 'predict'
        ]
    )
    
    args = train_args
    checkpoint_fname = 'checkpoint_last.pt'
    checkpoint_path = Path(args.save_dir) / checkpoint_fname
    y_preds = eval(args, False, checkpoint_path, None, args.dataset_name, predict_dataset)
    return y_preds

def graphormer_predict(pt_start, pt_end, dataset_name):
    convert_table_valid = pd.read_csv(GLOBAL_ROOT + '/datasets/' + dataset_name + '/raw/convert_roads_valid.csv').dropna()
    convert_table_valid['edge_coord_start'] = convert_table_valid['edge_coord_start'].apply(lambda x: json.loads(x))
    convert_table_valid['edge_coord_end'] = convert_table_valid['edge_coord_end'].apply(lambda x: json.loads(x))

    point_start = pt_start
    point_end = pt_end

    convert_table_valid['point_start_N'] = point_start[0]
    convert_table_valid['point_start_E'] = point_start[1]
    convert_table_valid['point_end_N'] = point_end[0]
    convert_table_valid['point_end_E'] = point_end[1]

    convert_table_valid['dist_start'] = convert_table_valid.apply(lambda x: (x['edge_coord_start'][0][0] - x['point_start_N'])**2 + (x['edge_coord_start'][0][1] - x['point_start_E'])**2, axis = 1)
    convert_table_valid['dist_end'] = convert_table_valid.apply(lambda x: (x['edge_coord_end'][0][0] - x['point_end_N'])**2 + (x['edge_coord_end'][0][1] - x['point_end_E'])**2, axis = 1)
    convert_table_valid['dist_mean'] = (convert_table_valid['dist_start'] + convert_table_valid['dist_end'])/2

    predict_table = convert_table_valid.sort_values(by = ['dist_mean']).reset_index(drop = True)[:1]

    dataset = single_geo_Abakan(predict_table)
    dataset = dataset.process()

    predicted_time = predict_time(dataset_name, dataset)

    return [predict_table['edges_geo'], predicted_time]