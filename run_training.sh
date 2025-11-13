#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py \
  --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --data_source "hotpotqa" \
  --max_samples 500 \
  --difficulty "hard" \
  --output_dir "./r-search-model" \
  --openai_api_key "your-openai-key" \
  --exa_api_key "your-exa-key" \
  --batch_size 1 \
  --max_steps 100 \
  --lr 5e-6 \
  --num_generations 4 \
  --gradient_accumulation_steps 4 \
  --max_new_tokens 1024 \
  --curriculum_steps 30 \
  --save_steps 50 \
  --use_wandb \
  --wandb_api_key "your-wandb-key" \
  --wandb_project "r-search-training" \
  --wandb_run_name "r-search-hotpotqa-500" \
  --push_to_hub \
  --hub_model_id "your-username/r-search-model" \
  --hub_token "your-hf-token" \
  --hub_private
