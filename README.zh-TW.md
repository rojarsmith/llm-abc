# LLM ABC

[English](README.md) | [繁體中文](README.zh-TW.md)

LLM ABC 是一個教學型專案，用來從零開始建立一個最小版的 ChatGPT 類系統。專案會先從 tiny PyTorch GPT 模型開始，接出 AI API 後端，並為後續 Next.js Web UI 做準備。

這個專案的目標不是一開始就做出強大的助理，而是幫助軟體開發者完整體驗以下路徑：

1. 隨機初始化模型可以產生 token，但還不會使用語言。
2. tiny 模型使用極小文字資料訓練後會出現什麼變化。
3. 使用更多資料或更多訓練步數後，模型效果如何改變。
4. 下載別人訓練好的 pretrained model 後，效果有什麼差異。
5. fine-tuned model 和原始 base model 的行為差異。

## 目前範圍

第一版只建立最小可用後端骨架：

- `packages/llm_core`：GPT 模型、tokenizer、文字生成邏輯。
- `apps/api`：FastAPI chat API，包含同步與非同步 job 端點。
- `scripts/smoke_chat.py`：不啟動伺服器也能檢查模型輸出的 smoke test。
- `docs/learning-experience.md`：第一階段學習閉環的英文驗證清單。

## 建立並啟用 Virtual Environment

本專案所有 Python 指令都應該在專案本地的 `.venv` 內執行。

以下指令請使用 Windows Command Prompt (`cmd.exe`)。`python` 指令必須指向 Windows CPython 3.11、3.12 或 3.13。先不要用 Python 3.14，因為目前這個 setup 下 PyTorch 沒有對應 wheel。

先確認 `cmd.exe` 會使用哪個 Python：

```cmd
where python
python --version
python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else '本專案請使用 Python 3.11、3.12 或 3.13')"
```

全新設定：

```cmd
python -m venv .venv
.venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip install -e . -r apps\api\requirements.txt
```

如果 `.venv` 是用錯的 Python 版本建立的，請重建：

```cmd
if defined VIRTUAL_ENV call deactivate
if exist .venv rmdir /s /q .venv

where python
python --version
python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else '本專案請使用 Python 3.11、3.12 或 3.13')"

python -m venv .venv
.venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip install -e . -r apps\api\requirements.txt
```

啟用後，`python` 和 `pip` 都應該指向 `.venv` 環境。

## 先驗證學習體驗

啟用 virtual environment 後，先跑 smoke test。它會載入一個完全未訓練的 tiny GPT 模型，並用同一個 prompt 產生文字接續。

```cmd
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 24
```

你應該觀察到：

1. 輸出不像真正回答，因為模型權重是隨機的。
2. prompt 會被轉成 token，再由模型一個 token 一個 token 接續產生。
3. 這個結果就是後續訓練前的 baseline。

試著改變生成參數：

```cmd
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 12
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 40 --temperature 1.0 --top-k 20
```

觀察重點：

- `max-new-tokens` 控制生成長度。
- `temperature=0` 會固定選最高分 token。
- `temperature>0` 會從機率分布抽樣。
- 未訓練模型即使會產生 token，也還沒有學會語言規律。

## 啟動 API

```cmd
python -m uvicorn apps.api.main:app --reload --port 8000
```

## 驗證同步 Chat API

以下範例在 Windows Command Prompt 使用 `curl`。

```cmd
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/models

curl -s -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Every effort moves you\",\"max_new_tokens\":24}"
```

## 驗證非同步 Chat Job

```cmd
for /f %i in ('curl -s -X POST http://127.0.0.1:8000/chat/jobs -H "Content-Type: application/json" -d "{\"message\":\"Every effort moves you\",\"max_new_tokens\":24}" ^| python -c "import sys,json; print(json.load(sys.stdin)['job_id'])"') do set JOB_ID=%i

curl -s "http://127.0.0.1:8000/chat/jobs/%JOB_ID%"
```

## 訓練 Tiny Model

執行命令列訓練 smoke test：

```cmd
python scripts\smoke_train.py --max-steps 80 --eval-every 10
```

或建立 API 訓練 job：

```cmd
for /f %i in ('curl -s -X POST http://127.0.0.1:8000/training/jobs -H "Content-Type: application/json" -d "{\"dataset_id\":\"every-effort\",\"base_model_id\":\"random-tiny-byte\",\"output_model_id\":\"trained-tiny-byte\",\"max_steps\":80,\"eval_every\":10,\"load_when_complete\":true}" ^| python -c "import sys,json; print(json.load(sys.stdin)['job_id'])"') do set TRAINING_JOB_ID=%i

curl -s "http://127.0.0.1:8000/training/jobs/%TRAINING_JOB_ID%"
```

job 成功後，可以用同一個 `/chat` prompt 比較 `random-tiny-byte` 和 `trained-tiny-byte`。

## 第一階段學習結論

這一版要先確認三件事：

1. 專案中自己寫的 GPT 架構可以被 API 呼叫。
2. 未訓練模型可以產生 token，但輸出還沒有語言能力。
3. API 已具備同步與非同步入口，下一步可以加入 training job，讓使用者比較訓練前後差異。

下一個建議里程碑是 `training/jobs`：用最短可用文字資料訓練 tiny model，儲存成 checkpoint，再回到 `/chat` 比較生成結果。

## 更多文件

- [Learning experience checklist](docs/learning-experience.md)
- [`smoke_chat.py` explained](docs/smoke-chat.md)
- [Minimal training loop](docs/training-loop.md)
- [繁體中文學習驗證清單](docs/learning-experience.zh-TW.md)
- [`smoke_chat.py` 繁體中文說明](docs/smoke-chat.zh-TW.md)
- [最小訓練閉環](docs/training-loop.zh-TW.md)
