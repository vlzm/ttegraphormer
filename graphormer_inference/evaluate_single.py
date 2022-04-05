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
from data_class import single_geo_Omsk, single_geo_Abakan, GraphormerPYGDataset_predict
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
from fairseq import models

import pathlib
link = pathlib.Path().resolve()
link = str(link).split('TransTTE')[0]
GLOBAL_ROOT = link + 'TransTTE'

sys.path.insert(2, GLOBAL_ROOT + '/graphormer')
from pretrain import load_pretrained_model
from data.pyg_datasets.pyg_dataset import GraphormerPYGDataset
from data.dataset import (
    BatchedDataDataset,
    TargetDataset,
    GraphormerDataset)

def eval(args, use_pretrained, checkpoint_path=None, logger=None, data_name = None):
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    seed = 71
    
    
    ### data loading
    print('START SINGLE')
    root = osp.join(GLOBAL_ROOT, 'datasets', data_name)
    if data_name == 'omsk':
        raw_dir = osp.join(root, 'processed', 'data_omsk_1')
        data = single_geo_Omsk(root = raw_dir)
    if data_name == 'abakan':
        raw_dir = osp.join(root, 'processed', 'data_abakan_1')
        data = single_geo_Abakan(root = raw_dir)
    print('END SINGLE')
    
    print('1')
    GPYG = GraphormerPYGDataset_predict(data, seed, None, data, data_name)
    print('2')
    batched_data = BatchedDataDataset(GPYG)
    print('3')
    data_sizes = np.array([128] * len(batched_data))
    print('4')
    dataset_total = NestedDictionaryDataset(
            {
                "nsamples": NumSamplesDataset(),
                "net_input": {"batched_data": batched_data},
                "target": batched_data,
            },
        sizes=data_sizes,
        )
    ###
    print('5')
    ### initialize task
    task = tasks.setup_task(cfg.task)
    print('6')
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
    print(y_pred)
    
    return y_pred

 


def main():
    parser = options.get_training_parser()
    parser.add_argument(
        "--split",
        type=str,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
    )
    parser.add_argument(
        "--metric",
        type=str,
    )
    parser.add_argument(
        "--mode",
        type=str,
    )
    
    args = options.parse_args_and_arch(parser)
    logger = logging.getLogger(__name__)
    if args.pretrained_model_name != "none":
        eval(args, True, logger=logger)
    elif hasattr(args, "save_dir"):
        checkpoint_fname = 'checkpoint_last.pt'
        checkpoint_path = Path(args.save_dir) / checkpoint_fname
        logger.info(f"evaluating checkpoint file {checkpoint_path}")
        eval(args, False, checkpoint_path, logger, args.dataset_name)



if __name__ == '__main__':
    main()

