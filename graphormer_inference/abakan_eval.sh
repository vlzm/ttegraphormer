#!/usr/bin/env bash

    
python evaluate_single.py \
    --user-dir ../../TransTTE/graphormer_repo/graphormer \
    --num-workers 10 \
    --ddp-backend=legacy_ddp \
    --dataset_name abakan \
    --dataset-source pyg \
    --num-atoms 6656 \
    --task graph_prediction \
    --criterion l1_loss \
    --arch graphormer_slim \
    --num-classes 1 \
    --batch-size 1 \
    --save-dir ../../TransTTE/graphormer_repo/examples/georides/abakan/ckpts/ \
    --split valid \
    --metric rmse \
    --mode predict