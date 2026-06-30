# Minimal Training Loop

[English](training-loop.md) | [繁體中文](training-loop.zh-TW.md)

This document explains the second learning loop in LLM ABC:

```text
random model -> tiny training data -> train -> save checkpoint -> load checkpoint -> chat again
```

The goal is not to train a useful assistant yet. The goal is to make training visible and testable.

## What Was Added

- `data/tiny/every-effort.txt`: a repeated tiny dataset for quick overfitting.
- `llm_core.training`: byte-token dataset, loss calculation, and training loop.
- `llm_core.checkpoints`: checkpoint save/list/load helpers.
- `scripts/smoke_train.py`: command-line training smoke test.
- `POST /training/jobs`: asynchronous API training job.
- `GET /training/jobs/{job_id}`: training job status and progress.
- `GET /checkpoints`: saved checkpoint list.
- `POST /models/load`: load a checkpoint as a chat model.

## Command-Line Smoke Test

Use Windows Command Prompt with `.venv` activated:

```cmd
python scripts\smoke_train.py --max-steps 80 --eval-every 10
```

For a code-level walkthrough, read [`smoke_train.py` explained](smoke-train.md).

Expected observations:

- `before training` looks random.
- Training progress prints loss values.
- Loss should generally move downward on this tiny dataset.
- A checkpoint is saved under `models\checkpoints`.
- `after training` should be less random than the baseline, because the model has started to overfit the repeated text.

## API Training Flow

Start the API:

```cmd
python -m uvicorn apps.api.main:app --reload --port 8000
```

Check available training datasets:

```cmd
curl -s http://127.0.0.1:8000/training/datasets
```

Create a training job:

```cmd
for /f %i in ('curl -s -X POST http://127.0.0.1:8000/training/jobs -H "Content-Type: application/json" -d "{\"dataset_id\":\"every-effort\",\"base_model_id\":\"random-tiny-byte\",\"output_model_id\":\"trained-tiny-byte\",\"max_steps\":80,\"eval_every\":10,\"load_when_complete\":true}" ^| python -c "import sys,json; print(json.load(sys.stdin)['job_id'])"') do set TRAINING_JOB_ID=%i
```

Check the training job until `status` becomes `succeeded`:

```cmd
curl -s "http://127.0.0.1:8000/training/jobs/%TRAINING_JOB_ID%"
```

List checkpoints:

```cmd
curl -s http://127.0.0.1:8000/checkpoints
```

## What a Checkpoint Contains

In this project, a checkpoint is a full model state snapshot, not a delta or accumulated patch.

Each checkpoint stores:

- model architecture config
- tokenizer name
- training summary
- full `model.state_dict()`

That means a checkpoint can be loaded directly as a complete trained model state for the same `GPTModel` implementation. It does not need previous checkpoints to be replayed first.

The loading flow is:

```text
create GPTModel from checkpoint config -> load full state_dict -> use it as trained-tiny-byte
```

This is different from LoRA adapters, delta weights, or patch checkpoints. Those formats store only differences relative to a base model. LLM ABC currently uses full checkpoints because they are easier to inspect and teach.

One important detail: the checkpoint does not contain the Python source code for `GPTModel`. The project still needs compatible model code and config when loading the checkpoint.

When `load_when_complete` is `true`, the trained checkpoint is automatically loaded as `trained-tiny-byte`. Compare the random model and trained model with the same prompt:

```cmd
curl -s -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"model_id\":\"random-tiny-byte\",\"message\":\"Every effort moves you\",\"max_new_tokens\":24}"

curl -s -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"model_id\":\"trained-tiny-byte\",\"message\":\"Every effort moves you\",\"max_new_tokens\":24}"
```

## Loading a Checkpoint Manually

If a checkpoint is saved but not loaded, use:

```cmd
curl -s -X POST http://127.0.0.1:8000/models/load ^
  -H "Content-Type: application/json" ^
  -d "{\"checkpoint_id\":\"YOUR_CHECKPOINT_ID\",\"model_id\":\"trained-tiny-byte\"}"
```

Then call `/chat` with `model_id` set to `trained-tiny-byte`.

## What You Should Learn

This stage demonstrates four ideas:

1. Training is just repeated prediction error reduction.
2. Loss gives a measurable signal while the model learns.
3. Tiny datasets make tiny models overfit quickly.
4. A full checkpoint is a complete trained model state that lets the API reuse trained weights later.

The next stage is available in [Minimal Web UI learning console](web-console.md).
