#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 fairseq-train \
--tensorboard-logdir ../../examples/stackoverflow/tensorlogs_stackover_v3 \
--user-dir ../../graphormer \
--num-workers 1 \
--ddp-backend=legacy_ddp \
--dataset-name stackoverflow \
--dataset-source pyg \
--num-atoms 1024 \
--task graph_prediction_link \
--criterion l1_loss_link_prediction \
--arch graphormer_slim \
--num-classes 1 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates 60000 --total-num-update 400000 \
--lr 2e-4 --end-learning-rate 1e-9 \
--batch-size 1 \
--data-buffer-size 20 \
--encoder-layers 12 \
--encoder-embed-dim 80 \
--encoder-ffn-embed-dim 80 \
--encoder-attention-heads 8 \
--max-epoch 10000 \
--save-dir ./ckpts