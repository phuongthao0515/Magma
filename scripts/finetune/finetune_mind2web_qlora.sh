#!/bin/bash
# Mind2Web QLoRA Finetuning Script
# Uses 4-bit quantization with LoRA for memory-efficient training

MODEL_PATH="microsoft/Magma-8B"
OUTPUT_DIR="./checkpoints/finetune-mind2web-qlora"

# Number of GPUs (adjust as needed)
NUM_GPUS=1

# For single GPU (recommended for QLoRA)
if [ "$NUM_GPUS" -eq 1 ]; then
    python train.py \
        --model_name_or_path $MODEL_PATH \
        --version magma_instruct \
        --data_path "data_configs/mind2web.yaml" \
        --vision_tower convnext_xxlarge \
        --img_size 512 \
        --max_num_crops 4 \
        --img_anyres_strategy crop \
        --vision_backbone "convnextxxlarge" \
        --is_multimodal True \
        --mm_projector_type mlp2x_gelu \
        --tune_mm_mlp_adapter True \
        --tune_vision_tokenizer 'none' \
        --mm_vision_select_layer -2 \
        --mm_use_image_start_end False \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "steps" \
        --eval_steps 500 \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 3 \
        --load_best_model_at_end True \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --learning_rate 2e-4 \
        --weight_decay 0.0 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --flash_attn_2_enabled True \
        --local_run False \
        --show_trace False \
        --run_name mind2web_qlora \
        --remove_static_trace_pts True \
        --bits 4 \
        --lora_enable True \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --double_quant True \
        --quant_type nf4
else
    # For multi-GPU with DeepSpeed ZeRO-2 (ZeRO-3 doesn't work well with quantization)
    torchrun --nproc_per_node=$NUM_GPUS train.py \
        --deepspeed ./trainer/deepspeed/zero2.json \
        --model_name_or_path $MODEL_PATH \
        --version magma_instruct \
        --data_path "data_configs/mind2web.yaml" \
        --vision_tower convnext_xxlarge \
        --img_size 512 \
        --max_num_crops 4 \
        --img_anyres_strategy crop \
        --vision_backbone "convnextxxlarge" \
        --is_multimodal True \
        --mm_projector_type mlp2x_gelu \
        --tune_mm_mlp_adapter True \
        --tune_vision_tokenizer 'none' \
        --mm_vision_select_layer -2 \
        --mm_use_image_start_end False \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "steps" \
        --eval_steps 500 \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 3 \
        --load_best_model_at_end True \
        --metric_for_best_model "eval_loss" \
        --greater_is_better False \
        --learning_rate 2e-4 \
        --weight_decay 0.0 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --flash_attn_2_enabled True \
        --local_run False \
        --show_trace False \
        --run_name mind2web_qlora \
        --remove_static_trace_pts True \
        --bits 4 \
        --lora_enable True \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --double_quant True \
        --quant_type nf4
fi
