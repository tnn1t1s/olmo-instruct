#!/bin/bash
# =============================================================================
# OLMo INIS/Bamboo DPO Training Script
# Uses AllenAI open-instruct: https://github.com/allenai/open-instruct
#
# Run AFTER train_sft.sh completes
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Start from SFT checkpoint
SFT_MODEL_PATH="${SFT_MODEL_PATH:-./outputs/olmo-inis-bamboo}"
OUTPUT_DIR="./outputs/olmo-inis-bamboo-dpo"
RUN_NAME="olmo-7b-inis-bamboo-dpo"

# Data paths
TRAIN_DATA="data/bamboo_dpo.jsonl,data/inis_dpo.jsonl"

# DPO hyperparameters
BETA=0.1
LEARNING_RATE=5e-7
NUM_EPOCHS=1
MAX_SEQ_LENGTH=4096
MAX_PROMPT_LENGTH=2048
BATCH_SIZE=1
GRAD_ACCUM=8
WARMUP_RATIO=0.1

# Hardware
NUM_GPUS=${NUM_GPUS:-1}

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
echo "=== OLMo INIS/Bamboo DPO Training ==="
echo "SFT Model: ${SFT_MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Beta: ${BETA}"
echo "GPUs: ${NUM_GPUS}"
echo ""

# Check if SFT model exists
if [ ! -d "$SFT_MODEL_PATH" ]; then
    echo "Error: SFT model not found at ${SFT_MODEL_PATH}"
    echo "Run train_sft.sh first to create the SFT checkpoint"
    exit 1
fi

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
# Launch DPO training
# -----------------------------------------------------------------------------
echo "Starting DPO training..."

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU with accelerate
    accelerate launch \
        --mixed_precision bf16 \
        --num_processes ${NUM_GPUS} \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_config.json \
        -m open_instruct.dpo_tune \
        --model_name_or_path ${SFT_MODEL_PATH} \
        --tokenizer_name ${SFT_MODEL_PATH} \
        --train_file ${TRAIN_DATA} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --max_prompt_length ${MAX_PROMPT_LENGTH} \
        --preprocessing_num_workers 8 \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LEARNING_RATE} \
        --lr_scheduler_type cosine \
        --warmup_ratio ${WARMUP_RATIO} \
        --weight_decay 0.01 \
        --num_train_epochs ${NUM_EPOCHS} \
        --beta ${BETA} \
        --output_dir ${OUTPUT_DIR} \
        --run_name ${RUN_NAME} \
        --logging_steps 10 \
        --save_strategy epoch \
        --save_total_limit 2 \
        --bf16 \
        --report_to wandb
else
    # Single GPU
    python -m open_instruct.dpo_tune \
        --model_name_or_path ${SFT_MODEL_PATH} \
        --tokenizer_name ${SFT_MODEL_PATH} \
        --train_file ${TRAIN_DATA} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --max_prompt_length ${MAX_PROMPT_LENGTH} \
        --preprocessing_num_workers 4 \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --learning_rate ${LEARNING_RATE} \
        --lr_scheduler_type cosine \
        --warmup_ratio ${WARMUP_RATIO} \
        --weight_decay 0.01 \
        --num_train_epochs ${NUM_EPOCHS} \
        --beta ${BETA} \
        --output_dir ${OUTPUT_DIR} \
        --run_name ${RUN_NAME} \
        --logging_steps 10 \
        --save_strategy epoch \
        --save_total_limit 2 \
        --bf16 \
        --report_to wandb
fi

echo ""
echo "=== DPO Training complete ==="
echo "Model saved to: ${OUTPUT_DIR}"
echo ""
echo "Training pipeline complete:"
echo "  1. SFT: ${SFT_MODEL_PATH}"
echo "  2. DPO: ${OUTPUT_DIR}"
