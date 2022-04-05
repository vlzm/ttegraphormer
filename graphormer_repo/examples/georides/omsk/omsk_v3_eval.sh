#!/usr/bin/env bash

    
python evaluate.py \
    --user-dir ../../../graphormer \
    --num-workers 1 \
    --ddp-backend=legacy_ddp \
    --dataset-name omsk \
    --dataset-source pyg \
    --task graph_prediction \
    --arch graphormer_slim \
    --num-classes 1 \
    --batch-size 64 \
    --save-dir .ckpts_v2/ \
    --split test \
    --metric l1_loss \
    --seed 1



