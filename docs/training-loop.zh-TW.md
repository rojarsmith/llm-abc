# 最小訓練閉環

[English](training-loop.md) | [繁體中文](training-loop.zh-TW.md)

這份文件說明 LLM ABC 的第二個學習閉環：

```text
隨機模型 -> tiny 訓練資料 -> 訓練 -> 儲存 checkpoint -> 載入 checkpoint -> 再聊天比較
```

這一階段的目標不是訓練出有用的助理，而是讓訓練這件事可以被看見、可以被測試。

## 新增內容

- `data/tiny/every-effort.txt`：重複的小資料集，用來快速 overfit。
- `llm_core.training`：byte-token dataset、loss 計算、訓練迴圈。
- `llm_core.checkpoints`：checkpoint 儲存、列出、載入工具。
- `scripts/smoke_train.py`：命令列訓練 smoke test。
- `POST /training/jobs`：非同步 API 訓練 job。
- `GET /training/jobs/{job_id}`：訓練 job 狀態與進度。
- `GET /checkpoints`：列出已儲存 checkpoint。
- `POST /models/load`：把 checkpoint 載入成可聊天模型。

## 命令列 Smoke Test

在 Windows Command Prompt 啟用 `.venv` 後執行：

```cmd
python scripts\smoke_train.py --max-steps 80 --eval-every 10
```

如果要看程式流程說明，請讀 [`smoke_train.py` 說明](smoke-train.zh-TW.md)。

應觀察：

- `before training` 看起來像隨機輸出。
- 訓練過程會印出 loss。
- 在這個 tiny dataset 上，loss 通常會往下降。
- checkpoint 會儲存在 `models\checkpoints`。
- `after training` 應該會比 baseline 少一點隨機感，因為模型開始 overfit 重複資料。

## API 訓練流程

啟動 API：

```cmd
python -m uvicorn apps.api.main:app --reload --port 8000
```

查看可用訓練資料：

```cmd
curl -s http://127.0.0.1:8000/training/datasets
```

建立訓練 job：

```cmd
for /f %i in ('curl -s -X POST http://127.0.0.1:8000/training/jobs -H "Content-Type: application/json" -d "{\"dataset_id\":\"every-effort\",\"base_model_id\":\"random-tiny-byte\",\"output_model_id\":\"trained-tiny-byte\",\"max_steps\":80,\"eval_every\":10,\"load_when_complete\":true}" ^| python -c "import sys,json; print(json.load(sys.stdin)['job_id'])"') do set TRAINING_JOB_ID=%i
```

查詢訓練 job，直到 `status` 變成 `succeeded`：

```cmd
curl -s "http://127.0.0.1:8000/training/jobs/%TRAINING_JOB_ID%"
```

列出 checkpoints：

```cmd
curl -s http://127.0.0.1:8000/checkpoints
```

## Checkpoint 裡有什麼

在目前這個專案裡，checkpoint 是完整模型狀態快照，不是差異檔，也不是累加 patch。

每個 checkpoint 會儲存：

- 模型架構 config
- tokenizer 名稱
- 訓練摘要
- 完整的 `model.state_dict()`

也就是說，checkpoint 可以直接被載入成同一個 `GPTModel` 實作下的完整訓練後模型狀態。它不需要先把前面的 checkpoint 依序重放。

載入流程是：

```text
根據 checkpoint config 建立 GPTModel -> 載入完整 state_dict -> 作為 trained-tiny-byte 使用
```

這和 LoRA adapter、delta weights、patch checkpoint 不同。那些格式通常只儲存相對於 base model 的差異。LLM ABC 目前使用 full checkpoint，因為它比較容易檢查，也比較適合教學。

有一點要注意：checkpoint 不包含 `GPTModel` 的 Python 原始碼。載入 checkpoint 時，專案仍然需要有相容的模型程式碼與 config。

當 `load_when_complete` 是 `true`，訓練完成的 checkpoint 會自動載入成 `trained-tiny-byte`。用同一個 prompt 比較隨機模型與訓練模型：

```cmd
curl -s -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"model_id\":\"random-tiny-byte\",\"message\":\"Every effort moves you\",\"max_new_tokens\":24}"

curl -s -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"model_id\":\"trained-tiny-byte\",\"message\":\"Every effort moves you\",\"max_new_tokens\":24}"
```

## 手動載入 Checkpoint

如果 checkpoint 已儲存但尚未載入，可以使用：

```cmd
curl -s -X POST http://127.0.0.1:8000/models/load ^
  -H "Content-Type: application/json" ^
  -d "{\"checkpoint_id\":\"YOUR_CHECKPOINT_ID\",\"model_id\":\"trained-tiny-byte\"}"
```

然後呼叫 `/chat`，並把 `model_id` 設成 `trained-tiny-byte`。

## 你應該學到什麼

這一階段展示四件事：

1. 訓練就是反覆降低預測錯誤。
2. loss 是模型學習過程中的可量測訊號。
3. tiny dataset 會讓 tiny model 很快 overfit。
4. full checkpoint 是完整的訓練後模型狀態，讓 API 之後可以重用訓練後權重。

下一階段已在 [最小 Web UI 學習控制台](web-console.zh-TW.md) 中提供。
