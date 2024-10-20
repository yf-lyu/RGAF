#!/usr/bin/env bash

CLIP_NAME="openai/clip-vit-large-patch14"
TEXT_PATH="twitter/dataset_text"
IMG_PATH="twitter/dataset_image"

CUDA_VISIBLE_DEVICES=1 python -u run_main.py \
          --clip_name=${CLIP_NAME} \
          --batch_size=16 \
          --pretrained_lr=3e-5 \
          --clip_lr=2e-6 \
          --other_lr=1e-4 \
          --num_epochs=10 \
          --warmup_ratio=0.1 \
          --warmup_epochs=10 \
          --do_train \
          --clip_knowledge \
          --device="cuda" \
          --seed=2023 \
          --embed_dim=768 \
          --dropout_prob=0.5 \
          --shared_space_dim=400 \
          --queue_size=4096 \
          --momentum=0.995 \
          --temp=0.07 \
          --alpha=0.5 \
          --ce_loss_weight=0.5 \
          --global_loss_weight=0.3 \
          --token_loss_weight=0.5 \
          --text_path=${TEXT_PATH} \
          --img_path=${IMG_PATH} \
          --save_model_path="model_ckpt_path"
