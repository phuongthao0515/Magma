#!/bin/bash
# Mind2Web QLoRA Training Script (Single GPU, Training Only)

python train.py \
    --model_name_or_path microsoft/Magma-8B \
    --data_path "data_configs/mind2web.yaml" \
    --output_dir ./checkpoints/finetune-mind2web-qlora \
    --is_multimodal True \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --save_steps 1 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --bits 4 \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --tune_mm_mlp_adapter True \
    --report_to none
