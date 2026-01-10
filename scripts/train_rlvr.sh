#!/bin/bash
# =============================================================================
# OLMo INIS/Bamboo RLVR Training Script
# Reinforcement Learning with Verifiable Rewards
# Uses AllenAI open-instruct: https://github.com/allenai/open-instruct
#
# Run AFTER train_dpo.sh completes
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Start from DPO checkpoint
DPO_MODEL_PATH="${DPO_MODEL_PATH:-./outputs/olmo-inis-bamboo-dpo}"
OUTPUT_DIR="./outputs/olmo-inis-bamboo-rlvr"
RUN_NAME="olmo-7b-inis-bamboo-rlvr"

# Data paths
TRAIN_DATA="data/bamboo_rlvr.jsonl,data/inis_rlvr.jsonl"

# RLVR/PPO hyperparameters
LEARNING_RATE=1e-6
KL_COEF=0.05
PPO_EPOCHS=4
CLIP_RANGE=0.2
NUM_SAMPLES=4
TEMPERATURE=0.7
MAX_SEQ_LENGTH=4096

# Hardware
NUM_GPUS=${NUM_GPUS:-1}

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
echo "=== OLMo INIS/Bamboo RLVR Training ==="
echo "DPO Model: ${DPO_MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "KL Coef: ${KL_COEF}"
echo "GPUs: ${NUM_GPUS}"
echo ""

# Check if DPO model exists
if [ ! -d "$DPO_MODEL_PATH" ]; then
    echo "Error: DPO model not found at ${DPO_MODEL_PATH}"
    echo "Run train_dpo.sh first to create the DPO checkpoint"
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
# Note on verification
# -----------------------------------------------------------------------------
# RLVR requires a verification function to provide rewards.
# Our verifier is in evaluation/verifier.py::compute_reward
#
# For integration with open-instruct, you may need to:
# 1. Register the verifier as a custom reward function
# 2. Or use their built-in math verifier and adapt our data format
#
# See: https://github.com/allenai/open-instruct/tree/main/scripts/train/rlvr
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Launch RLVR training
# -----------------------------------------------------------------------------
echo "Starting RLVR training..."
echo ""
echo "NOTE: This script provides the configuration for RLVR training."
echo "The exact command depends on your open-instruct version."
echo "See the open-instruct RLVR documentation for the current API."
echo ""

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU RLVR training
    # Typically uses vLLM for inference and multiple GPUs for training
    accelerate launch \
        --mixed_precision bf16 \
        --num_processes ${NUM_GPUS} \
        -m open_instruct.ppo_vllm_thread \
        --model_name_or_path ${DPO_MODEL_PATH} \
        --tokenizer_name ${DPO_MODEL_PATH} \
        --dataset_name ${TRAIN_DATA} \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --learning_rate ${LEARNING_RATE} \
        --kl_coef ${KL_COEF} \
        --ppo_epochs ${PPO_EPOCHS} \
        --cliprange ${CLIP_RANGE} \
        --num_samples ${NUM_SAMPLES} \
        --temperature ${TEMPERATURE} \
        --output_dir ${OUTPUT_DIR} \
        --run_name ${RUN_NAME} \
        --logging_steps 10 \
        --save_steps 500 \
        --bf16 \
        --report_to wandb \
        --reward_model_path "evaluation.verifier:compute_reward"
else
    # Single GPU (limited but possible for small models)
    echo "Single GPU RLVR training is not recommended."
    echo "RLVR typically requires separate GPUs for inference and training."
    echo ""
    echo "Consider using at least 2 GPUs with:"
    echo "  NUM_GPUS=2 ./scripts/train_rlvr.sh"
    echo ""
    echo "Or use the debug script from open-instruct:"
    echo "  scripts/train/debug/single_gpu_on_beaker.sh"
    exit 1
fi

echo ""
echo "=== RLVR Training complete ==="
echo "Model saved to: ${OUTPUT_DIR}"
echo ""
echo "Full training pipeline complete:"
echo "  1. SFT:  ./outputs/olmo-inis-bamboo"
echo "  2. DPO:  ./outputs/olmo-inis-bamboo-dpo"
echo "  3. RLVR: ${OUTPUT_DIR}"
