# OLMo INIS/Bamboo Fine-tuning

Fine-tuning OLMo for two domains:
- **INIS**: Technical/structured content (nuclear science, extraction, Q&A)
- **Bamboo**: Declarative reasoning with Ibis (deferred execution, data operations)

## Training Pipeline

Follows the OLMo 3 post-training recipe:

```
Base Model → SFT → DPO → RLVR
```

| Stage | Script | Data | Description |
|-------|--------|------|-------------|
| SFT | `scripts/train_sft.sh` | `*_train.jsonl` | Supervised fine-tuning |
| DPO | `scripts/train_dpo.sh` | `*_dpo.jsonl` | Direct preference optimization |
| RLVR | `scripts/train_rlvr.sh` | `*_rlvr.jsonl` | RL with verifiable rewards |

## Training Data Format

### SFT Data

Uses [Tulu 3 message format](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture):

```json
{
  "id": "example_001",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "source": "domain-name"
}
```

**Bamboo domain** uses `<think>` tags for reasoning traces (OLMo-Think style):

```json
{"role": "assistant", "content": "<think>\nIbis uses deferred execution...\n```python\nimport ibis\n...\n```\n</think>\n\nHere's the solution: ..."}
```

### DPO Data

Preference pairs with prompt, chosen, and rejected:

```json
{
  "id": "dpo_001",
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "correct response"}],
  "rejected": [{"role": "assistant", "content": "incorrect response"}],
  "source": "domain-dpo"
}
```

### RLVR Data

Prompts with ground truth for verification:

```json
{
  "id": "rlvr_001",
  "prompt": "Using Ibis, calculate total revenue...",
  "ground_truth": "2500",
  "verification_type": "numeric",
  "test_data": {"sales": [...]},
  "source": "bamboo-rlvr"
}
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run evaluation on training data
python evaluation/ibis_evaluator.py data/bamboo_train.jsonl bamboo
python evaluation/ibis_evaluator.py data/inis_train.jsonl inis

# Training (requires open-instruct)
pip install git+https://github.com/allenai/open-instruct.git

# Stage 1: SFT
./scripts/train_sft.sh

# Stage 2: DPO (after SFT completes)
./scripts/train_dpo.sh

# Stage 3: RLVR (after DPO completes, requires multi-GPU)
NUM_GPUS=2 ./scripts/train_rlvr.sh
```

## Project Structure

```
olmo-instruct/
├── data/
│   ├── bamboo_train.jsonl    # SFT: Ibis reasoning examples
│   ├── bamboo_dpo.jsonl      # DPO: preference pairs
│   ├── bamboo_rlvr.jsonl     # RLVR: verifiable prompts
│   ├── inis_train.jsonl      # SFT: Technical Q&A
│   ├── inis_dpo.jsonl        # DPO: preference pairs
│   └── inis_rlvr.jsonl       # RLVR: verifiable prompts
├── configs/
│   ├── sft_config.yaml       # SFT reference config
│   ├── dpo_config.yaml       # DPO reference config
│   ├── rlvr_config.yaml      # RLVR reference config
│   └── ds_config.json        # DeepSpeed config
├── evaluation/
│   ├── __init__.py
│   ├── ibis_evaluator.py     # Evaluation harness
│   └── verifier.py           # RLVR reward functions
├── scripts/
│   ├── train_sft.sh          # SFT training
│   ├── train_dpo.sh          # DPO training
│   └── train_rlvr.sh         # RLVR training
└── pyproject.toml
```

## Evaluation & Verification

### Post-training Evaluation

```bash
python evaluation/ibis_evaluator.py data/bamboo_train.jsonl bamboo
```

### RLVR Verification

The verifier provides binary rewards (0/1) for RL training:

```python
from evaluation import compute_reward

reward = compute_reward(model_output, {
    "ground_truth": "2500",
    "verification_type": "numeric",
    "test_data": {...}
})
```

Verification types:
- `numeric`: Compare extracted number to ground truth
- `string`: Exact string match
- `json_field`: Check specific JSON field value

## Configuration

### SFT (`configs/sft_config.yaml`)

| Parameter | Value |
|-----------|-------|
| Base model | `allenai/OLMo-2-1124-7B` |
| Learning rate | `2e-5` |
| Epochs | 3 |

### DPO (`configs/dpo_config.yaml`)

| Parameter | Value |
|-----------|-------|
| Base model | SFT checkpoint |
| Learning rate | `5e-7` |
| Beta | 0.1 |
| Epochs | 1 |

### RLVR (`configs/rlvr_config.yaml`)

| Parameter | Value |
|-----------|-------|
| Base model | DPO checkpoint |
| Learning rate | `1e-6` |
| KL coef | 0.05 |
| PPO epochs | 4 |

## References

- [OLMo 3 Blog](https://allenai.org/blog/olmo3)
- [Tulu 3 Technical](https://allenai.org/blog/tulu-3-technical)
- [open-instruct](https://github.com/allenai/open-instruct)
- [Ibis Framework](https://ibis-project.org/)
- [RLVR Explained](https://www.promptfoo.dev/blog/rlvr-explained/)
