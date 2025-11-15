#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python train.py \
  --name irra_card0_size64_AttentionPooling_lrfacor8.0_targetlr1e-8_60epoch_CEsmooth0.1 \
  --dataset_name CUHK-PEDES \
  --root_dir /home/oqh/dataset \
  --output_dir /data/oqh_data/outputs/IRRA_decouple \
  --batch_size 64 \
  --img_aug \
  --sampler random \
  --MLM \
  --loss_names sdm+mlm+id \
  --num_epoch 60 \
  --lr_factor 8.0 \
  --target_lr 1e-8