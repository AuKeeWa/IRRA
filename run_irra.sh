#!/bin/bash
# DATASET_NAME="CUHK-PEDES"
DATASET_NAME="ICFG-PEDES"

CUDA_VISIBLE_DEVICES=4 \
python train.py \
--name irra \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60