# LLM ABC

[English](README.md) | [繁體中文](README.zh-TW.md)

LLM ABC is an educational project for building a minimal ChatGPT-like system from scratch. It starts with a tiny PyTorch GPT model, exposes it through an AI API backend, and prepares the path for a Next.js Web UI.

The goal is not to build a powerful assistant immediately. The goal is to help software developers experience the full learning path:

1. A randomly initialized model that can generate tokens but cannot use language well.
2. A tiny model trained on very small text data.
3. A model trained on more data or for more steps.
4. A downloaded pretrained model.
5. A fine-tuned model compared against the original base model.

## Current Scope

This first version builds the smallest usable backend skeleton:

- `packages/llm_core`: GPT model, tokenizer, and text generation logic.
- `apps/api`: FastAPI chat API with synchronous and asynchronous job endpoints.
- `apps/web`: Next.js learning console for chat, training, checkpoints, and comparison.
- `scripts/smoke_chat.py`: A no-server smoke test for the model output.
- `docs/learning-experience.md`: A guided checklist for validating the first learning loop.

## Create and Activate the Virtual Environment

All Python commands in this project should run inside the project-local `.venv`.

Use Windows Command Prompt (`cmd.exe`) for the commands below. The `python` command must point to Windows CPython 3.11, 3.12, or 3.13. Do not use Python 3.14 for now, because PyTorch wheels are not available for that version in this setup.

Check which Python executable `cmd.exe` will use:

```cmd
where python
python --version
python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 'Use Python 3.11, 3.12, or 3.13 for this project')"
```

For a fresh setup:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip install -e . -r apps\api\requirements.txt
```

If you created `.venv` with the wrong Python version, recreate it:

```cmd
if defined VIRTUAL_ENV call deactivate
if exist .venv rmdir /s /q .venv

where python
python --version
python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 'Use Python 3.11, 3.12, or 3.13 for this project')"

python -m venv .venv
.venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip install -e . -r apps\api\requirements.txt
```

After activation, `python` and `pip` should resolve to the `.venv` environment.

## Validate the Learning Experience First

After the virtual environment is activated, run the smoke test. It loads a completely untrained tiny GPT model and generates a continuation from the same prompt.

```cmd
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 24
```

You should observe:

1. The output does not look like a real answer because the model weights are random.
2. The prompt is converted into tokens, then the model generates one token at a time.
3. This result is the baseline before any training.

Try changing generation parameters:

```cmd
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 12
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 40 --temperature 1.0 --top-k 20
```

What to observe:

- `max-new-tokens` controls the generated length.
- `temperature=0` always picks the highest-scoring token.
- `temperature>0` samples from the probability distribution.
- An untrained model can generate tokens, but it has not learned language patterns yet.

## Start the API

```cmd
python -m uvicorn apps.api.main:app --reload --port 8000
```

## Validate the Synchronous Chat API

These examples use `curl` in Windows Command Prompt.

```cmd
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/models

curl -s -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Every effort moves you\",\"max_new_tokens\":24}"
```

## Validate the Asynchronous Chat Job

```cmd
for /f %i in ('curl -s -X POST http://127.0.0.1:8000/chat/jobs -H "Content-Type: application/json" -d "{\"message\":\"Every effort moves you\",\"max_new_tokens\":24}" ^| python -c "import sys,json; print(json.load(sys.stdin)['job_id'])"') do set JOB_ID=%i

curl -s "http://127.0.0.1:8000/chat/jobs/%JOB_ID%"
```

## Train the Tiny Model

Run the command-line training smoke test:

```cmd
python scripts\smoke_train.py --max-steps 80 --eval-every 10
```

Or start an API training job:

```cmd
for /f %i in ('curl -s -X POST http://127.0.0.1:8000/training/jobs -H "Content-Type: application/json" -d "{\"dataset_id\":\"every-effort\",\"base_model_id\":\"random-tiny-byte\",\"output_model_id\":\"trained-tiny-byte\",\"max_steps\":80,\"eval_every\":10,\"load_when_complete\":true}" ^| python -c "import sys,json; print(json.load(sys.stdin)['job_id'])"') do set TRAINING_JOB_ID=%i

curl -s "http://127.0.0.1:8000/training/jobs/%TRAINING_JOB_ID%"
```

After the job succeeds, compare `random-tiny-byte` and `trained-tiny-byte` with the same `/chat` prompt.

## Start the Web Learning Console

Keep the API running, then open a second Windows Command Prompt:

```cmd
cd apps\web
npm install
npm run dev
```

Open `http://127.0.0.1:3000` and use the console to chat, train, load checkpoints, and compare model outputs.

## First-Stage Learning Conclusion

This version should help you confirm three things:

1. A GPT architecture written in this project can be called by an API.
2. An untrained model can generate tokens, but the output has no language ability yet.
3. The API already has synchronous and asynchronous entry points, so the next step can add training jobs and compare results before and after training.

The recommended next milestone is `training/jobs`: train a tiny model on the shortest useful text dataset, save it as a checkpoint, then return to `/chat` and compare the generated output.

## More Documentation

- [Learning experience checklist](docs/learning-experience.md)
- [`smoke_chat.py` explained](docs/smoke-chat.md)
- [Minimal training loop](docs/training-loop.md)
- [`smoke_train.py` explained](docs/smoke-train.md)
- [Minimal Web UI learning console](docs/web-console.md)
- [`smoke_train.py` 繁體中文說明](docs/smoke-train.zh-TW.md)
- [繁體中文學習驗證清單](docs/learning-experience.zh-TW.md)
- [`smoke_chat.py` 繁體中文說明](docs/smoke-chat.zh-TW.md)
- [最小訓練閉環](docs/training-loop.zh-TW.md)
- [最小 Web UI 學習控制台](docs/web-console.zh-TW.md)
