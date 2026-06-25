# `smoke_chat.py` 說明

[English](smoke-chat.md) | [繁體中文](smoke-chat.zh-TW.md)

這份文件解釋 `scripts/smoke_chat.py` 在做什麼，以及它為什麼存在。

這支腳本是本專案的第一個學習檢查點。它讓你不用啟動 FastAPI、不用建立 Web UI、也不用先訓練模型，就能直接跑一次模型核心。

## 為什麼需要這支腳本

`smoke_chat.py` 只回答一個很小但很重要的問題：

我們本地寫的 GPT 模型程式，能不能接收文字、轉成 token、跑過模型、再產生新的 token？

在這個階段，好的結果不是聰明回答。好的結果是確認整條推論路徑可以跑通：

```text
message -> prompt -> token ids -> GPTModel -> generated token ids -> text
```

因為模型是隨機初始化，所以輸出看起來應該會很隨機。這就是後續訓練完成後要拿來比較的 baseline。

## 如何執行

請使用 Windows Command Prompt (`cmd.exe`)。先建立並啟用專案本地的 `.venv`：

```shell
where python
python --version
python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else '本專案請使用 Python 3.11、3.12 或 3.13')"

python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e . -r apps\api\requirements.txt
```

如果 `python --version` 顯示 Python 3.14，請調整 Windows PATH，讓 `python` 指向 Python 3.11、3.12 或 3.13。

接著執行：

```cmd
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 24
```

輸出大致會長這樣：

```text
model_id: random-tiny-byte
device: cuda
parameters: 136,704
prompt_tokens: 39
tokens_generated: 24

prompt:
User: Every effort moves you
Assistant:

generated continuation:
'?\\xc37A\\xef\\xecR...'

learning note:
This model is randomly initialized. It can generate tokens, but it has not learned language yet.
```

## 如果看到 `actions-runner` 警告

你可能會看到類似這樣的警告：

```text
UserWarning: Failed to initialize NumPy: No module named 'numpy'
... C:\actions-runner\_work\pytorch\pytorch\...
```

`actions-runner` 不是代表你的電腦正在跑 GitHub Actions。它只是 PyTorch 預編譯 wheel 裡保留下來的建置機器來源路徑。

真正需要處理的是 `No module named 'numpy'`。請在 `.venv` 內重新安裝專案依賴：

```cmd
python -m pip install -e . -r apps\api\requirements.txt
```

專案現在已把 `numpy` 加入依賴，重新安裝後這個警告應該會消失。

## 程式流程

### 1. 讓本地 package 可以被 import

腳本會把 `packages/llm_core` 加進 Python import path：

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "llm_core"))
```

這讓腳本在專案還沒有正式安裝成套件前，也能直接 import `llm_core`。

### 2. 讀取命令列參數

腳本支援：

```text
--message
--model-id
--max-new-tokens
--temperature
--top-k
```

這些參數讓你不用改程式碼，就能改 prompt 和生成行為。

### 3. 載入模型設定

```python
cfg = MODEL_CONFIGS[args.model_id]
```

預設 `model_id` 是 `random-tiny-byte`。它目前定義了一個很小的 GPT：

```text
vocab_size: 257
context_length: 64
emb_dim: 64
n_heads: 4
n_layers: 2
```

這個模型刻意做得很小，讓第一個學習閉環能在本機快速跑起來。

### 4. 建立 tokenizer

```python
tokenizer = ByteTokenizer()
```

第一版使用 `ByteTokenizer`，不是 `tiktoken`。

這樣可以讓第一個學習閉環少一點外部依賴：

- UTF-8 bytes 會變成 token ids `0..255`。
- token id `256` 保留給 EOS。
- vocabulary size 是 `257`。

這個 tokenizer 不是最終 GPT-2 相容 tokenizer。它是第一個後端閉環用的教學工具。

### 5. 固定隨機種子

```python
torch.manual_seed(cfg.seed)
```

模型權重仍然是隨機的，但隨機初始化會變得可重現。這讓 smoke test 的結果更容易比較。

### 6. 選擇 CPU 或 GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

如果有 CUDA，就使用 GPU。否則使用 CPU。

### 7. 建立 GPT 模型

```python
model = GPTModel(cfg.to_dict()).to(device)
model.eval()
```

這一步會從零建立 tiny GPT 架構。它沒有載入訓練好的權重。

`model.eval()` 會把模型切到推論模式。這很重要，因為生成時應該關閉 dropout。

### 8. 建立聊天格式 prompt

```python
prompt = prepare_chat_prompt(args.message)
```

如果 message 是：

```text
Every effort moves you
```

prompt 會變成：

```text
User: Every effort moves you
Assistant:
```

這個簡單格式是在為後續 Web Chat API 的形狀做準備。

### 9. 把文字轉成 token ids

```python
input_ids = tokenizer.encode(prompt)
idx = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
```

tokenizer 會把文字轉成整數。tensor shape 會變成：

```text
[batch_size, sequence_length]
```

單一 prompt 的 `batch_size` 是 `1`。

### 10. 產生新的 tokens

```python
output = generate(
    model=model,
    idx=idx,
    max_new_tokens=args.max_new_tokens,
    context_size=cfg.context_length,
    temperature=args.temperature,
    top_k=args.top_k,
    eos_id=tokenizer.eos_id,
)
```

`generate()` 會重複做以下步驟：

1. 把 context 裁切到模型支援的長度。
2. 跑 GPT 模型。
3. 取最後一個 token 位置的 logits。
4. 選出下一個 token。
5. 把這個 token 接到序列後面。

這就是 autoregressive language model 的核心生成迴圈。

### 11. 只解碼新產生的 continuation

```python
output_ids = output.squeeze(0).tolist()
generated_ids = output_ids[len(input_ids) :]
reply = tokenizer.decode(generated_ids)
```

完整輸出包含原本的 prompt 加上新產生的 tokens。腳本會把 prompt 部分拿掉，所以 `reply` 只會顯示模型在 `Assistant:` 後面產生的內容。

### 12. 印出學習用資訊

腳本會印出：

- `model_id`：使用哪一個模型設定。
- `device`：CPU 或 CUDA。
- `parameters`：可訓練參數量。
- `prompt_tokens`：prompt 被轉成幾個 input tokens。
- `tokens_generated`：產生幾個新 tokens。
- `generated continuation`：解碼後的新文字。

生成文字會用 `ascii(reply)` 印出，因為未訓練的 byte-level 模型可能產生任意 byte，有些字元不一定能在每個終端機正常顯示。

## 參數說明

| 參數 | 預設值 | 意義 |
|---|---:|---|
| `--message` | `Every effort moves you` | 放進 prompt 的使用者訊息。 |
| `--model-id` | `random-tiny-byte` | `MODEL_CONFIGS` 裡的模型設定 key。 |
| `--max-new-tokens` | `24` | 要產生幾個新的 token。 |
| `--temperature` | `0.0` | `0` 表示固定選最高分 token；越高越偏向隨機抽樣。 |
| `--top-k` | `None` | 如果有設定，只允許分數最高的前 `k` 個 token 參與抽樣。 |

## 你應該學到什麼

這支腳本最重要的學習點，是把「模型結構可以運作」和「模型已經學會語言」分開看。

目前程式已具備產生 token 的結構：

- tokenizer
- embeddings
- transformer blocks
- output head
- autoregressive generation loop

但模型還沒有學到語言行為，因為權重是隨機的。

所以這個 smoke test 應該在訓練前先跑。後續加入訓練後，你可以用同一個 prompt 再跑一次，直接比較訓練前後的行為差異。

## 相關檔案

- [`scripts/smoke_chat.py`](../scripts/smoke_chat.py)
- [`packages/llm_core/llm_core/configs.py`](../packages/llm_core/llm_core/configs.py)
- [`packages/llm_core/llm_core/tokenizer.py`](../packages/llm_core/llm_core/tokenizer.py)
- [`packages/llm_core/llm_core/model.py`](../packages/llm_core/llm_core/model.py)
- [`packages/llm_core/llm_core/generation.py`](../packages/llm_core/llm_core/generation.py)
