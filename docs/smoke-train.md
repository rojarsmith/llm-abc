# `smoke_train.py` Explained

[English](smoke-train.md) | [繁體中文](smoke-train.zh-TW.md)

This document explains the command-line training smoke test:

```cmd
python scripts\smoke_train.py --max-steps 80 --eval-every 10
```

Run it only after the project `.venv` is activated.

The examples below use Windows Command Prompt (`cmd.exe`) and the `python` command from `.venv`.

The script demonstrates the second learning checkpoint in LLM ABC:

```text
random model -> generate before sample -> train on tiny text -> generate after sample -> save full checkpoint
```

The goal is not to create a useful assistant. The goal is to make the training loop visible, short, and easy to compare against the earlier untrained model.

## Quick Run

Use this when you want the normal learning experience:

```cmd
python scripts\smoke_train.py --max-steps 80 --eval-every 10
```

Use this when you only want to verify that the code path works:

```cmd
python scripts\smoke_train.py --max-steps 10 --eval-every 5
```

Expected output shape:

```text
base_model_id: random-tiny-byte
output_model_id: trained-tiny-byte
device: cpu

before training:
'...random-looking text...'

training progress:
step 0010/0080 loss=...
step 0020/0080 loss=...

after training:
'...less random or more dataset-like text...'

checkpoint_id: trained-tiny-byte-...
checkpoint_path: ...\models\checkpoints\...
```

The generated text may still look rough. That is expected. The model is tiny and the dataset is intentionally small.

## Step-by-Step Code Flow

### 1. Make the Local Package Importable

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "llm_core"))
```

The script can be run directly from the repository root without installing a published package. It adds `packages/llm_core` to Python's import path so `llm_core` can be imported.

### 2. Import the Training Building Blocks

```python
from llm_core.checkpoints import save_checkpoint
from llm_core.configs import MODEL_CONFIGS
from llm_core.model import GPTModel
from llm_core.tokenizer import ByteTokenizer
from llm_core.training import TrainingConfig, generate_sample, train_tiny_language_model
```

These imports separate the learning pieces:

- `GPTModel`: the tiny GPT architecture.
- `ByteTokenizer`: turns UTF-8 bytes into token IDs.
- `TrainingConfig`: the small set of training knobs.
- `generate_sample`: produces comparable before/after text.
- `train_tiny_language_model`: runs the training loop.
- `save_checkpoint`: writes a full model state snapshot.

### 3. Parse Command-Line Arguments

```python
parser.add_argument("--dataset", default=str(ROOT / "data" / "tiny" / "every-effort.txt"))
parser.add_argument("--base-model-id", default="random-tiny-byte")
parser.add_argument("--output-model-id", default="trained-tiny-byte")
parser.add_argument("--max-steps", type=int, default=80)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--block-size", type=int, default=32)
parser.add_argument("--learning-rate", type=float, default=3e-3)
parser.add_argument("--eval-every", type=int, default=10)
parser.add_argument("--sample-prompt", default="Every effort moves you")
```

Important arguments:

| Argument | Meaning |
| --- | --- |
| `--dataset` | Text file used for the tiny training loop. |
| `--base-model-id` | Which model config to start from. |
| `--output-model-id` | Name to give the trained checkpoint when loading it later. |
| `--max-steps` | Number of optimizer updates. Higher values make overfitting easier to see. |
| `--batch-size` | Number of text windows per training step. |
| `--block-size` | Number of input tokens in each training window. |
| `--learning-rate` | AdamW step size. |
| `--eval-every` | How often progress is printed. |
| `--sample-prompt` | Prompt used for before/after comparison. |

### 4. Build an Untrained Model

```python
cfg = MODEL_CONFIGS[args.base_model_id]
tokenizer = ByteTokenizer()
torch.manual_seed(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(cfg.to_dict()).to(device)
model.eval()
```

At this point the model has architecture and random weights, but it has not learned from text yet.

`torch.manual_seed(cfg.seed)` makes the random initialization repeatable, which helps when comparing learning runs.

### 5. Generate the Baseline Sample

```python
before_sample = generate_sample(
    model=model,
    tokenizer=tokenizer,
    prompt=args.sample_prompt,
    device=device,
    context_size=cfg.context_length,
    max_new_tokens=24,
)
```

This is the "before training" output. It gives you a baseline for the same prompt before any optimizer step has happened.

### 6. Create the Training Config

```python
training_config = TrainingConfig(
    max_steps=args.max_steps,
    batch_size=args.batch_size,
    block_size=args.block_size,
    learning_rate=args.learning_rate,
    eval_every=args.eval_every,
    sample_prompt=args.sample_prompt,
    seed=cfg.seed,
)
```

This keeps the script small and makes the same settings reusable by the API training job.

### 7. Read the Tiny Dataset

```python
text = Path(args.dataset).read_text(encoding="utf-8")
```

The default dataset is `data\tiny\every-effort.txt`. It is deliberately repetitive, so the tiny model can overfit quickly. That makes learning visible in a short run.

### 8. Train the Model

```python
summary = train_tiny_language_model(
    model=model,
    tokenizer=tokenizer,
    text=text,
    device=device,
    config=training_config,
    progress_callback=lambda event: print(
        f"step {event['step']:04d}/{event['max_steps']:04d} "
        f"loss={event['loss']:.4f} tokens_seen={event['tokens_seen']}"
    ),
)
```

Inside `train_tiny_language_model`, the training loop repeatedly:

1. slices text into byte-token windows,
2. asks the model to predict the next token,
3. calculates cross-entropy loss,
4. runs backpropagation,
5. updates weights with AdamW,
6. reports progress every `eval_every` steps.

The loss is the main learning signal. On this tiny repeated dataset, it should generally move downward, although individual printed values can still fluctuate.

### 9. Attach Baseline Metadata

```python
summary["before_sample"] = before_sample
summary["dataset_path"] = str(Path(args.dataset))
```

The checkpoint records both the baseline sample and the dataset path. This makes the saved checkpoint easier to explain later.

### 10. Save a Full Checkpoint

```python
checkpoint = save_checkpoint(
    checkpoint_dir=ROOT / "models" / "checkpoints",
    model=model,
    model_id=args.output_model_id,
    base_model_id=args.base_model_id,
    model_config=cfg,
    tokenizer_name="byte",
    training_summary=summary,
)
```

In this project, a checkpoint is a full model state snapshot. It stores the complete `model.state_dict()`, not only a delta from the base model.

That means the checkpoint can be loaded directly as a trained model state, as long as the same compatible `GPTModel` code and config are available.

### 11. Print the Learning Result

```python
print("after training:")
print(ascii(summary["sample_text"]))
print(f"checkpoint_id: {checkpoint['checkpoint_id']}")
print(f"checkpoint_path: {checkpoint['path']}")
```

The script prints:

- the generated text before training,
- loss progress during training,
- the generated text after training,
- the checkpoint ID and path.

`ascii(...)` is used because the byte-level random output can contain control characters or invalid-looking byte sequences. Escaping them keeps the terminal output readable.

## What You Should Learn

This script teaches the smallest complete training loop:

1. A randomly initialized model gives a baseline.
2. Training uses text tokens to reduce prediction error.
3. Loss is the visible measurement of learning.
4. A tiny repeated dataset makes overfitting easy to observe.
5. A full checkpoint turns the trained weights into a reusable model state.

After this script works, the API training job is easier to understand because it uses the same core training pieces asynchronously.

## Related Files

- `scripts/smoke_train.py`: command-line training smoke test.
- `packages/llm_core/llm_core/training.py`: dataset, loss, optimizer loop, and sample generation.
- `packages/llm_core/llm_core/checkpoints.py`: full checkpoint save/load helpers.
- `data/tiny/every-effort.txt`: tiny training dataset.
- `docs/training-loop.md`: API and checkpoint learning loop.
