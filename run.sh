#!/bin/bash
# Word SoM QLoRA Training Script
#
# Usage:
#   bash run.sh                          # from base Magma-8B
#   bash run.sh --from-mind2web          # from mind2web-merged base
#   bash run.sh --cycle                  # with UPS cycling (3h train, 1h rest)
#   bash run.sh --from-mind2web --cycle  # both flags

TRAIN_HOURS=3
REST_HOURS=1
CYCLE_MODE=false

# Default: train from original Magma-8B
BASE_MODEL="microsoft/Magma-8B"
OUTPUT_DIR="./checkpoints/finetune-3apps-r32-a64-maxlen2560-focal-marks-5actions"
RUN_NAME="3apps-5actions-r32-a64-focal-marks-2560"

# Parse flags
for arg in "$@"; do
    case $arg in
        --from-mind2web)
            BASE_MODEL="./checkpoints/magma-8b-mind2web-merged"
            OUTPUT_DIR="./checkpoints/finetune-wordsom-from-mind2web-qlora-768-img-size"
            RUN_NAME="word-som-from-mind2web-qlora"
            ;;
        --cycle)
            CYCLE_MODE=true
            ;;
    esac
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Base model:  $BASE_MODEL"
echo "Output dir:  $OUTPUT_DIR"
echo "Cycle mode:  $CYCLE_MODE"

run_training() {
    if [ "$CYCLE_MODE" = true ]; then
        timeout ${TRAIN_HOURS}h python train.py "$@"
    else
        python train.py "$@"
    fi
}

TRAIN_ARGS=(
    --model_name_or_path $BASE_MODEL
    --data_path "data_configs/office_3apps_5actions.yaml"
    --output_dir $OUTPUT_DIR
    --is_multimodal True
    --bf16 True
    --num_train_epochs 2
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 4
    --learning_rate 5e-5
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --save_steps 100
    --logging_steps 10
    --evaluation_strategy no
    --gradient_checkpointing True
    --bits 4
    --lora_enable True
    --lora_r 32
    --lora_alpha 64
    --tune_mm_mlp_adapter True
    --img_size 768
    --model_max_length 2560
    --flash_attn_2_enabled True
    --dataloader_num_workers 4
    --report_to wandb
    --run_name "$RUN_NAME"
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