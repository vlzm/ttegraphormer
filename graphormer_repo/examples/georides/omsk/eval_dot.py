import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys
from os import path
from data_class import geo_Omsk
import os
import os.path as osp
from graphormer.data.wrapper import preprocess_item
from graphormer.data.collator import collator_graphormer
import torch
from graphormer.pretrain import load_pretrained_model
from pathlib import Path
from fairseq import checkpoint_utils, utils, options, tasks
from graphormer.tasks.graph_prediction import GraphPredictionTask

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
sys.path.insert(1, '/home/jovyan/graphormer_v2/')
from graphormer.pretrain import load_pretrained_model
import logging

def main():
    checkpoint_folder = 'ckpts_v2'
    checkpoint_fname = 'checkpoint62.pt'
    save_dir = osp.join(checkpoint_folder, checkpoint_fname)
    split = 'train'
    checkpoint_path = Path(save_dir) 
    root = osp.join('dataset', 'omsk')
    raw_dir = osp.join(root, 'processed', 'data_omsk_1')
    update_dir = osp.join(root, 'processed', 'data_omsk_1_upd')
    split = 'train'
    raw_data = geo_Omsk(raw_dir, split = split)
    data = preprocess_item(raw_data[0])
    model_state = torch.load(checkpoint_path)["model"]
    task = GraphPredictionTask()
    model = task.build_model('graphormer')
    model.load_state_dict(model_state, strict=True, model_cfg=cfg.model)
    with torch.no_grad():
        model.eval()
        sample = utils.move_to_cuda(sample)
        y = model(**data)[:, 0, :].reshape(-1)
    print(y)    



if __name__ == '__main__':
    main()