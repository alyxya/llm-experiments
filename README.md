# llm-experiments

A repo for iterating on LLM training and architecture experiments. Shared infrastructure (tokenizer, model, trainer) makes it trivial to add new experiments without touching shared code.

## Setup

```bash
uv sync
```

## Experiments

### Sequence Copying

Validates that a small transformer can learn to copy input sequences — a fundamental test of attention mechanics.

- ~610K parameter GPT (3 layers, 4 heads, 128 embed dim)
- Character-level tokenizer over `abcdefghij`
- Task: `<bos> input <sep> input <eos>` with loss masked to output tokens only
- Reaches 95%+ exact-match accuracy by step 5000

```bash
python -m experiments.copying
```

Takes ~30s on MPS, ~2-3 min on CPU. After training, drops into an interactive REPL where you can type strings and see if the model copies them.

Training loss is configurable via `TrainConfig.loss`:
- `cross_entropy` (default)
- `sse_onehot_logits` (sum of squared error on logits vs one-hot targets, with PAD masked out)

## Adding a New Experiment

1. Create `experiments/<name>.py`
2. Define a config dataclass, `get_batch` closure, `eval_fn` closure, and `main()`
3. Import shared `model.py`, `tokenizer.py`, `trainer.py`
4. Run with `python -m experiments.<name>`

Each training run saves a timestamped directory under `runs/` with config, loss log, and checkpoints.
