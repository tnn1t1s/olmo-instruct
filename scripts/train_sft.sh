#!/bin/bash
# =============================================================================
# OLMo INIS/Bamboo SFT Training Script
# Uses AllenAI open-instruct: https://github.com/allenai/open-instruct
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME="allenai/OLMo-2-1124-7B"
OUTPUT_DIR="./outputs/olmo-inis-bamboo"
RUN_NAME="olmo-7b-inis-bamboo-sft"

# Data paths (relative to repo root)
TRAIN_DATA="data/bamboo_train.jsonl,data/inis_train.jsonl"

# Training hyperparameters
LEARNING_RATE=2e-5
NUM_EPOCHS=3
MAX_SEQ_LENGTH=4096
BATCH_SIZE=1
GRAD_ACCUM=8
WARMUP_RATIO=0.03

# Hardware (adjust based on your setup)
NUM_GPUS=${NUM_GPUS:-1}

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
echo "=== OLMo INIS/Bamboo SFT Training ==="
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo ""

# Check if open-instruct is installed
if ! python -c "import open_instruct" 2>/dev/null; then
    echo "Error: open-instruct not installed"
    echo "Install with: pip install git+https://github.com/allenai/open-instruct.git"
    exit 1
fi

# Check data files exist
for f in $(echo $TRAIN_DATA | tr ',' ' '); do
    if [ ! -f "$f" ]; then
        echo "Error: Training file not found: $f"
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Prepare dataset for open-instruct format
# -----------------------------------------------------------------------------
# open-instruct expects HuggingFace dataset format or specific JSONL structure
# Our data is already in Tulu 3 format (messages array)

# -----------------------------------------------------------------------------
# Launch training
# -----------------------------------------------------------------------------
echo "Starting training..."

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU with accelerate
    accelerate launch \
        --mixed_precision bf16 \
        --num_processes ${NUM_GPUS} \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_config.json \
        -m open_instruct.finetune \
        --model_name_or_path ${MODEL_NAME} \
        --tokenizer_name ${MODEL_NAME} \
        --train_file ${TRAIN_DATA} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --preprocessing_num_workers 8 \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LEARNING_RATE} \
        --lr_scheduler_type cosine \
        --warmup_ratio ${WARMUP_RATIO} \
        --weight_decay 0.01 \
        --num_train_epochs ${NUM_EPOCHS} \
        --output_dir ${OUTPUT_DIR} \
        --run_name ${RUN_NAME} \
        --logging_steps 10 \
        --save_strategy epoch \
        --save_total_limit 3 \
        --bf16 \
        --report_to wandb
else
    # Single GPU
    python -m open_instruct.finetune \
        --model_name_or_path ${MODEL_NAME} \
        --tokenizer_name ${MODEL_NAME} \
        --train_file ${TRAIN_DATA} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --preprocessing_num_workers 4 \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LEARNING_RATE} \
        --lr_scheduler_type cosine \
        --warmup_ratio ${WARMUP_RATIO} \
        --weight_decay 0.01 \
        --num_train_epochs ${NUM_EPOCHS} \
        --output_dir ${OUTPUT_DIR} \
        --run_name ${RUN_NAME} \
        --logging_steps 10 \
        --save_strategy epoch \
        --save_total_limit 3 \
        --bf16 \
        --report_to wandb
fi

echo ""
echo "=== Training complete ==="
echo "Model saved to: ${OUTPUT_DIR}"
