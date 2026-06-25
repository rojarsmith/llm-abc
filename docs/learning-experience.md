# Minimal Backend Learning Checklist

[English](learning-experience.md) | [繁體中文](learning-experience.zh-TW.md)

This checklist validates the first-stage learning experience. The goal is not to get a smart answer yet. The goal is to confirm that a model can generate tokens, while an untrained model has not learned language.

For the code-level walkthrough, read [`smoke_chat.py` explained](smoke-chat.md). For the next training loop, read [Minimal training loop](training-loop.md).

## 0. Use the Project Virtual Environment

All commands in this checklist assume the project-local `.venv` is activated.

Use Windows Command Prompt (`cmd.exe`) for the commands below. The `python` command must point to Windows CPython 3.11, 3.12, or 3.13, not Python 3.14.

Check which Python executable `cmd.exe` will use:

```cmd
where python
python --version
python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 'Use Python 3.11, 3.12, or 3.13 for this project')"
```

Create and activate the venv:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip install -e . -r apps\api\requirements.txt
```

## 1. Core Model Smoke Test

Run:

```cmd
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 24
```

Expected observations:

- `parameters` shows the tiny GPT parameter count. It is currently 136,704.
- `prompt_tokens` shows how many tokens the tokenizer produced from the prompt.
- `tokens_generated` shows how many tokens the model generated.
- `generated continuation` will look like escaped random bytes. This is correct because the model has not been trained.

Learning point:

An untrained model is not unable to run. It can run, but it has not learned a language distribution.

## 2. Change the Output Length

Run:

```cmd
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 8
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 40
```

Expected observations:

- Larger `max-new-tokens` values produce longer continuations.
- The content is still unreliable because length does not make the model smarter.

Learning point:

Inference parameters control generation behavior. They do not replace training.

## 3. Change the Sampling Strategy

Run:

```cmd
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 24 --temperature 1.0 --top-k 20
```

Expected observations:

- `temperature=0` always picks the highest-scoring token.
- `temperature>0` samples from the probability distribution.
- For an untrained model, sampling usually just makes the random output more varied.

Learning point:

Temperature and top-k are generation strategies, not sources of knowledge.

## 4. Validate the API Entry Points

Start the API:

```cmd
python -m uvicorn apps.api.main:app --reload --port 8000
```

Health check:

```cmd
curl -s http://127.0.0.1:8000/health
```

Model list:

```cmd
curl -s http://127.0.0.1:8000/models
```

Synchronous chat:

```cmd
curl -s -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Every effort moves you\",\"max_new_tokens\":8}"
```

Expected observations:

- The API returns `model_id`, `prompt`, `reply`, `prompt_tokens`, and `tokens_generated`.
- `reply` is still the untrained model baseline.

Learning point:

A future Next.js UI can now send user input to this PyTorch LLM backend.

## 5. Validate the Asynchronous Job Flow

Create a job:

```cmd
for /f %i in ('curl -s -X POST http://127.0.0.1:8000/chat/jobs -H "Content-Type: application/json" -d "{\"message\":\"Every effort moves you\",\"max_new_tokens\":8}" ^| python -c "import sys,json; print(json.load(sys.stdin)['job_id'])"') do set JOB_ID=%i
```

Check the job:

```cmd
curl -s "http://127.0.0.1:8000/chat/jobs/%JOB_ID%"
```

Expected observations:

- The job moves from `queued` or `running` to `succeeded`.
- `result` contains a generated response similar to `/chat`.

Learning point:

LLM question answering can be wrapped as a job. Training and fine-tuning can later use the same pattern.

## 6. First-Stage Completion Criteria

The completion criteria are not that the answer looks like ChatGPT. They are:

- The model core runs.
- The API can call the model.
- Both synchronous and asynchronous entry points work.
- The learner can clearly see the untrained baseline.

The next stage is now available: train the tiny model on a small dataset, save a checkpoint, then return to `/chat` and compare outputs before and after training. See [Minimal training loop](training-loop.md).
