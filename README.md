# OLMo INIS/Bamboo Fine-tuning

Fine-tuning OLMo for two domains:
- **INIS**: Technical/structured content (nuclear science, extraction, Q&A)
- **Bamboo**: Declarative reasoning with Ibis (deferred execution, data operations)

## Training Pipeline

Follows the OLMo 3 post-training recipe:

```
Base Model → SFT → DPO → [RLVR]
```

| Stage | Script | Data |
|-------|--------|------|
| SFT | `scripts/train_sft.sh` | `*_train.jsonl` |
| DPO | `scripts/train_dpo.sh` | `*_dpo.jsonl` |

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

**INIS domain** uses structured outputs (JSON extraction, Q&A):

```json
{"role": "assistant", "content": "```json\n{\"isotope\": \"Cs-137\", ...}\n```"}
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
```

## Project Structure

```
olmo-instruct/
├── data/
│   ├── bamboo_train.jsonl    # SFT: Ibis reasoning examples
│   ├── bamboo_dpo.jsonl      # DPO: preference pairs
│   ├── inis_train.jsonl      # SFT: Technical Q&A examples
│   └── inis_dpo.jsonl        # DPO: preference pairs
├── configs/
│   ├── sft_config.yaml       # SFT reference config
│   ├── dpo_config.yaml       # DPO reference config
│   └── ds_config.json        # DeepSpeed config
├── evaluation/
│   ├── __init__.py
│   └── ibis_evaluator.py     # Evaluation harness
├── scripts/
│   ├── train_sft.sh          # SFT training script
│   └── train_dpo.sh          # DPO training script
└── pyproject.toml
```

## Evaluation

The evaluation harness validates:

**Bamboo domain:**
- Code extraction from `<think>` blocks
- Ibis code syntax validity
- Execution on DuckDB backend

**INIS domain:**
- JSON format validity
- Structure compliance

```bash
python evaluation/ibis_evaluator.py data/bamboo_train.jsonl bamboo
```

## Configuration

### SFT (`configs/sft_config.yaml`)

| Parameter | Value |
|-----------|-------|
| Base model | `allenai/OLMo-2-1124-7B` |
| Learning rate | `2e-5` |
| Epochs | 3 |
| Max sequence length | 4096 |

### DPO (`configs/dpo_config.yaml`)

| Parameter | Value |
|-----------|-------|
| Base model | SFT checkpoint |
| Learning rate | `5e-7` |
| Beta | 0.1 |
| Epochs | 1 |

## References

- [OLMo 3 Blog](https://allenai.org/blog/olmo3)
- [Tulu 3 Technical](https://allenai.org/blog/tulu-3-technical)
- [open-instruct](https://github.com/allenai/open-instruct)
- [Ibis Framework](https://ibis-project.org/)
