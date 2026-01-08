# OLMo INIS/Bamboo Fine-tuning

Fine-tuning OLMo for two domains:
- **INIS**: Technical/structured content (nuclear science, extraction, Q&A)
- **Bamboo**: Declarative reasoning with Ibis (deferred execution, data operations)

## Training Data Format

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

### Bamboo Domain

Outputs include `<think>` tags for reasoning traces (OLMo-Think style):

```json
{"role": "assistant", "content": "<think>\nIbis uses deferred execution...\n\n```python\nimport ibis\n...\n```\n</think>\n\nHere's the solution: ..."}
```

### INIS Domain

Structured outputs (JSON extraction, Q&A, summarization):

```json
{"role": "assistant", "content": "```json\n{\"isotope\": \"Cs-137\", ...}\n```"}
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run evaluation on training data
python evaluation/ibis_evaluator.py data/bamboo_train.jsonl bamboo
python evaluation/ibis_evaluator.py data/inis_train.jsonl inis

# Train (requires open-instruct)
pip install git+https://github.com/allenai/open-instruct.git
./scripts/train_sft.sh
```

## Project Structure

```
olmo-instruct/
├── data/
│   ├── bamboo_train.jsonl    # Ibis reasoning examples
│   └── inis_train.jsonl      # Technical Q&A examples
├── configs/
│   ├── sft_config.yaml       # Reference configuration
│   └── ds_config.json        # DeepSpeed config
├── evaluation/
│   ├── __init__.py
│   └── ibis_evaluator.py     # Evaluation harness
├── scripts/
│   └── train_sft.sh          # Training script
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

See `configs/sft_config.yaml` for hyperparameters. Key settings:

| Parameter | Value |
|-----------|-------|
| Base model | `allenai/OLMo-2-1124-7B` |
| Learning rate | `2e-5` |
| Epochs | 3 |
| Max sequence length | 4096 |

## References

- [OLMo 3 Blog](https://allenai.org/blog/olmo3)
- [Tulu 3 Technical](https://allenai.org/blog/tulu-3-technical)
- [open-instruct](https://github.com/allenai/open-instruct)
- [Ibis Framework](https://ibis-project.org/)
