#!/bin/bash
# Mind2Web Evaluation Script
# Evaluate checkpoints and select the best one

CHECKPOINT_DIR="./checkpoints/finetune-mind2web-qlora"
DATA_PATH="datasets/mind2web"
OUTPUT_DIR="results/mind2web_eval"
MAX_SAMPLES=100  # Set to smaller number for quick testing, remove for full eval

# Evaluate all checkpoints
python scripts/evaluation/eval_mind2web.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --base_model ./models/Magma-8B \
    --data $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --max_samples $MAX_SAMPLES \
    --eval_all \
    --use_4bit \
    --save_images

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Check checkpoint_ranking.json for best checkpoint"
