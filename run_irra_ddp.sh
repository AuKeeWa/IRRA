#!/bin/bash
# DDP version for 4 GPUs
# export CUDA_VISIBLE_DEVICES=4,5,6,7
CUDA_VISIBLE_DEVICES=4 \
python train.py \
  --name irra_ddp_1gpu_on_cn2_card4 \
  --dataset_name CUHK-PEDES \
  --root_dir /media/data5/zhangquan/oqh/dataset/T2I \
  --output_dir /media/data5/zhangquan/oqh/outputs/IRRA_decouple \
  --sampler identity \
  --batch_size 64 \
  --img_aug \
  --MLM \
  --loss_names sdm+mlm+id \
  --num_epoch 60 \
  --decouple \
  --id_head_layers 1 
