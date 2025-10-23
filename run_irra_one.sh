#!/bin/bash

CUDA_VISIBLE_DEVICES=5 \
python train.py \
  --name irra_cn3_card5_layers2_2_regloss0.01_cosine_random_addinit_idlossW0.01_size64_minusIDfeat \
  --dataset_name CUHK-PEDES \
  --root_dir /media/data5/zhangquan/oqh/dataset/T2I \
  --output_dir /media/data5/zhangquan/oqh/outputs/IRRA_decouple \
  --batch_size 64 \
  --img_aug \
  --sampler random \
  --MLM \
  --loss_names sdm+mlm+id \
  --num_epoch 180 \
  --decouple  \
  --id_loss_weight 0.01 \
  --id_head_layers 2 \
  --id_pred_layers 2 \
  --reg_loss_weight 0.01 \
  --reg_loss_type cosine