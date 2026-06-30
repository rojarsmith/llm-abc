# 最小 Web UI 學習控制台

[English](web-console.md) | [繁體中文](web-console.zh-TW.md)

這份文件說明 LLM ABC 的第一個 Web UI 里程碑。

這個控制台會把小型 Next.js app 接到既有 FastAPI 後端，讓學習閉環可以在瀏覽器中操作：

```text
用隨機模型聊天 -> 訓練 tiny model -> 觀察 loss -> 載入 checkpoint -> 比較輸出
```

## 新增內容

- `apps/web`：最小 Next.js 學習控制台。
- Chat view：把 prompt 送到指定模型。
- Compare view：用同一個 prompt 比較兩個模型。
- Training view：建立非同步 training job 並觀察進度。
- Checkpoints view：列出 full checkpoint，並載入成可聊天模型。
- API CORS 支援本機瀏覽器開發。

## 啟動 API

請使用 Windows Command Prompt，並先啟用 `.venv`：

```cmd
python -m uvicorn apps.api.main:app --reload --port 8000
```

## 啟動 Web UI

開第二個 Windows Command Prompt：

```cmd
cd apps\web
npm install
npm run dev
```

然後開啟：

```text
http://127.0.0.1:3000
```

UI 預設使用這個 API URL：

```text
http://127.0.0.1:8000
```

如果要在啟動前修改 API URL，可以在目前這個 Command Prompt 先設定：

```cmd
set NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
npm run dev
```

## 學習流程

1. 開啟 Chat tab。
2. 用 `random-tiny-byte` 送出 `Every effort moves you`。
3. 開啟 Training tab。
4. 使用 `every-effort`、`80` steps、`trained-tiny-byte` 建立 training job。
5. 觀察 loss 與 tokens 的變化。
6. job 成功後回到 Chat。
7. 用同一個 prompt 比較 `random-tiny-byte` 與 `trained-tiny-byte`。

## 為何這階段重要

CLI 與 API 已經證明訓練能跑。Web UI 會把同一個後端變成可操作的學習介面：

- 模型狀態看得見。
- 訓練進度看得見。
- checkpoints 看得見。
- 訓練前/後行為看得見。

下一階段要加入更大資料集、下載 pretrained model、fine-tuning 比較時，可以沿用這個 UI 結構。
