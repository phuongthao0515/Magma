#!/bin/bash
# Mind2Web QLoRA Finetuning Script
# Uses 4-bit quantization with LoRA for memory-efficient training
# Includes automatic evaluation after training

MODEL_PATH="microsoft/Magma-8B"
OUTPUT_DIR="./checkpoints/finetune-mind2web-qlora"
DATA_PATH="datasets/mind2web"
EVAL_OUTPUT_DIR="results/mind2web_eval"
EVAL_MAX_SAMPLES=200  # Number of samples for evaluation (set to 0 to skip eval)

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

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""

# Run evaluation on all checkpoints
if [ "$EVAL_MAX_SAMPLES" -gt 0 ]; then
    echo "=========================================="
    echo "Running Evaluation on Checkpoints..."
    echo "=========================================="

    python scripts/evaluation/eval_mind2web.py \
        --checkpoint_dir $OUTPUT_DIR \
        --base_model $MODEL_PATH \
        --data $DATA_PATH \
        --output_dir $EVAL_OUTPUT_DIR \
        --max_samples $EVAL_MAX_SAMPLES \
        --eval_all \
        --use_4bit

    echo ""
    echo "=========================================="
    echo "Evaluation Complete!"
    echo "=========================================="
    echo "Results saved to: $EVAL_OUTPUT_DIR"
    echo "Check $EVAL_OUTPUT_DIR/checkpoint_ranking.json for best checkpoint"
else
    echo "Skipping evaluation (EVAL_MAX_SAMPLES=0)"
    echo "To evaluate manually, run:"
    echo "  python scripts/evaluation/eval_mind2web.py --checkpoint_dir $OUTPUT_DIR --eval_all"
fi
