# Minimal Web UI Learning Console

[English](web-console.md) | [繁體中文](web-console.zh-TW.md)

This document explains the first Web UI milestone for LLM ABC.

The console connects a small Next.js app to the existing FastAPI backend and makes the learning loop visible in a browser:

```text
chat with random model -> train tiny model -> watch loss -> load checkpoint -> compare outputs
```

## What Was Added

- `apps/web`: a minimal Next.js learning console.
- Chat view: send a prompt to a selected model.
- Compare view: run the same prompt against two models.
- Training view: start an asynchronous training job and watch progress.
- Checkpoints view: list saved full checkpoints and load one as a chat model.
- API CORS support for local browser development.

## Run the API

Use Windows Command Prompt with `.venv` activated:

```cmd
python -m uvicorn apps.api.main:app --reload --port 8000
```

## Run the Web UI

Open a second Windows Command Prompt:

```cmd
cd apps\web
npm install
npm run dev
```

Then open:

```text
http://127.0.0.1:3000
```

The UI uses this API URL by default:

```text
http://127.0.0.1:8000
```

To override it for the current Command Prompt before starting the web app:

```cmd
set NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
npm run dev
```

## Learning Flow

1. Open the Chat tab.
2. Send `Every effort moves you` to `random-tiny-byte`.
3. Open the Training tab.
4. Start a training job with `every-effort`, `80` steps, and `trained-tiny-byte`.
5. Watch loss and tokens update while the job runs.
6. Return to Chat after the job succeeds.
7. Compare `random-tiny-byte` and `trained-tiny-byte` with the same prompt.

## Why This Stage Matters

The CLI and API already prove that training works. The Web UI turns the same backend into a learning surface:

- model state is visible,
- training progress is visible,
- checkpoints are visible,
- before/after behavior is visible.

This makes the next stages easier: larger datasets, downloaded pretrained models, and fine-tuning comparisons can reuse the same UI structure.
