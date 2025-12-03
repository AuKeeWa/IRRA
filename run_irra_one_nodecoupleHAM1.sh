#!/bin/bash

CUDA_VISIBLE_DEVICES=1 \
python train.py \
  --name irra_card1_size128_lr5e-6_lrfactor10_targetlr0_60epoch_CEsmooth0.0_temperature2.0_IDlossweight1.0_HAMpretrain_CMT3ID3Gated_ \
  --dataset_name CUHK-PEDES \
  --root_dir /home/oqh/dataset \
  --output_dir /data/oqh_data/outputs/IRRA_decouple \
  --batch_size 128 \
  --img_aug \
  --sampler random \
  --MLM \
  --loss_names sdm+mlm+id \
  --num_epoch 60 \
  --lr 5e-6 \
  --lr_factor 10.0 \
  --target_lr 0 \
  --id_temperature 2.0 \
  --id_label_smoothing 0.0 \
  --id_loss_weight 1.0 \
  --pretrain_ckpt_file '/data/oqh_data/random100w_2HAMcaptions/best0.pth' \
  --use_gated_cmt \
  --use_id_gate