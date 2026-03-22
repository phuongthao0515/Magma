#!/bin/bash
# Word SoM QLoRA Training Script
#
# Usage:
#   bash run.sh              # normal run (no cycling)
#   bash run.sh --cycle      # auto-cycle: train 3h, rest 1h, repeat

TRAIN_HOURS=3
REST_HOURS=1
OUTPUT_DIR="./checkpoints/finetune-wordsom-qlora-768-img-size"
CYCLE_MODE=false

if [ "$1" == "--cycle" ]; then
    CYCLE_MODE=true
    echo "Cycle mode: train ${TRAIN_HOURS}h, rest ${REST_HOURS}h"
fi

run_training() {
    if [ "$CYCLE_MODE" = true ]; then
        timeout ${TRAIN_HOURS}h python train.py "$@"
    else
        python train.py "$@"
    fi
}

TRAIN_ARGS=(
    --model_name_or_path microsoft/Magma-8B
    --data_path "data_configs/word_som.yaml"
    --output_dir $OUTPUT_DIR
    --is_multimodal True
    --bf16 True
    --num_train_epochs 1
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 4
    --learning_rate 1e-4
    --save_steps 100
    --logging_steps 10
    --evaluation_strategy steps
    --eval_steps 100
    --gradient_checkpointing True
    --bits 4
    --lora_enable True
    --lora_r 64
    --lora_alpha 16
    --tune_mm_mlp_adapter True
    --img_size 768
    --report_to wandb
    --run_name "word-som-qlora"
)

if [ "$CYCLE_MODE" = true ]; then
    while true; do
        echo "$(date): Starting training for ${TRAIN_HOURS}h..."
        run_training "${TRAIN_ARGS[@]}"
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "$(date): Training completed!"
            break
        fi
        echo "$(date): Resting for ${REST_HOURS}h..."
        sleep ${REST_HOURS}h
    done
else
    run_training "${TRAIN_ARGS[@]}"
fi
