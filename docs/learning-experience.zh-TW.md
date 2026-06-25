# 最小後端學習驗證清單

[English](learning-experience.md) | [繁體中文](learning-experience.zh-TW.md)

這份清單用來驗證第一階段的學習體驗。目標不是立刻得到聰明回答，而是確認「模型可以產生 token，但未訓練模型還沒有學會語言」這件事能被開發者清楚體驗。

如果要看程式流程說明，請讀 [`smoke_chat.py` 說明](smoke-chat.zh-TW.md)。下一階段訓練閉環請讀 [最小訓練閉環](training-loop.zh-TW.md)。

## 0. 使用專案 Virtual Environment

這份清單中的所有指令，都假設已經啟用專案本地的 `.venv`。

以下指令請使用 Windows Command Prompt (`cmd.exe`)。`python` 指令必須指向 Windows CPython 3.11、3.12 或 3.13，先不要用 Python 3.14。

先確認 `cmd.exe` 會使用哪個 Python：

```cmd
where python
python --version
python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else '本專案請使用 Python 3.11、3.12 或 3.13')"
```

建立並啟用 venv：

```cmd
python -m venv .venv
.venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip install -e . -r apps\api\requirements.txt
```

## 1. 核心模型 smoke test

執行：

```cmd
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 24
```

應觀察：

- `parameters` 會顯示 tiny GPT 的參數量，目前是 136,704。
- `prompt_tokens` 表示 prompt 被 tokenizer 轉成幾個 token。
- `tokens_generated` 表示模型接續產生幾個 token。
- `generated continuation` 會像 escaped random bytes。這是正確現象，因為模型還沒訓練。

學習重點：

未訓練模型不是不能跑，而是可以跑但還沒有學到語言分布。

## 2. 改變輸出長度

執行：

```cmd
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 8
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 40
```

應觀察：

- `max-new-tokens` 越大，生成 continuation 越長。
- 內容仍然不可靠，因為這只改變生成長度，不會讓模型變聰明。

學習重點：

推論參數只能控制生成行為，不能取代訓練。

## 3. 改變抽樣方式

執行：

```cmd
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 24 --temperature 1.0 --top-k 20
```

應觀察：

- `temperature=0` 會固定選最高分 token。
- `temperature>0` 會從機率分布抽樣。
- 對未訓練模型來說，抽樣通常只是讓隨機輸出更不固定。

學習重點：

temperature 和 top-k 是生成策略，不是知識來源。

## 4. 驗證 API 入口

先啟動 API：

```cmd
python -m uvicorn apps.api.main:app --reload --port 8000
```

健康檢查：

```cmd
curl -s http://127.0.0.1:8000/health
```

模型清單：

```cmd
curl -s http://127.0.0.1:8000/models
```

同步聊天：

```cmd
curl -s -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Every effort moves you\",\"max_new_tokens\":8}"
```

應觀察：

- API 會回傳 `model_id`、`prompt`、`reply`、`prompt_tokens`、`tokens_generated`。
- `reply` 仍是未訓練模型的 baseline。

學習重點：

現在 Next.js UI 已經可以把使用者輸入送到自己的 PyTorch LLM 後端。

## 5. 驗證非同步 job

建立 job：

```cmd
for /f %i in ('curl -s -X POST http://127.0.0.1:8000/chat/jobs -H "Content-Type: application/json" -d "{\"message\":\"Every effort moves you\",\"max_new_tokens\":8}" ^| python -c "import sys,json; print(json.load(sys.stdin)['job_id'])"') do set JOB_ID=%i
```

查詢 job：

```cmd
curl -s "http://127.0.0.1:8000/chat/jobs/%JOB_ID%"
```

應觀察：

- job 會從 `queued` 或 `running` 進到 `succeeded`。
- `result` 中會包含和 `/chat` 類似的生成結果。

學習重點：

LLM 問答可以被包成 job。後續訓練與 fine-tuning 也會沿用同一種模式。

## 6. 第一階段完成標準

完成標準不是「回答看起來像 ChatGPT」，而是：

- 模型核心可執行。
- API 可呼叫模型。
- 同步與非同步入口都可用。
- 使用者能清楚看到未訓練 baseline。

下一階段訓練閉環已可使用：用小資料訓練 tiny model，儲存 checkpoint，再回到 `/chat` 比較訓練前後差異。請參考 [最小訓練閉環](training-loop.zh-TW.md)。
