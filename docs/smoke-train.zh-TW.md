# `smoke_train.py` 說明

[English](smoke-train.md) | [繁體中文](smoke-train.zh-TW.md)

這份文件說明命令列訓練 smoke test：

```cmd
python scripts\smoke_train.py --max-steps 80 --eval-every 10
```

請先啟用專案本地的 `.venv`，再執行這個指令。

以下範例使用 Windows Command Prompt (`cmd.exe`) 與 `.venv` 內的 `python` 指令。

這支腳本展示 LLM ABC 的第二個學習檢查點：

```text
隨機模型 -> 產生訓練前樣本 -> 用 tiny text 訓練 -> 產生訓練後樣本 -> 儲存 full checkpoint
```

目標不是訓練出有用的助理，而是讓訓練迴圈變得看得見、跑得短、可以和前面的未訓練模型比較。

## 快速執行

一般學習體驗使用：

```cmd
python scripts\smoke_train.py --max-steps 80 --eval-every 10
```

只想確認流程可跑通時使用：

```cmd
python scripts\smoke_train.py --max-steps 10 --eval-every 5
```

預期輸出形狀：

```text
base_model_id: random-tiny-byte
output_model_id: trained-tiny-byte
device: cpu

before training:
'...看起來隨機的文字...'

training progress:
step 0010/0080 loss=...
step 0020/0080 loss=...

after training:
'...比較不隨機，或比較像資料集的文字...'

checkpoint_id: trained-tiny-byte-...
checkpoint_path: ...\models\checkpoints\...
```

產生的文字仍然可能很粗糙，這是正常的。模型很小，資料集也刻意保持很小。

## 程式流程逐段說明

### 1. 讓本地 package 可以被 import

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "llm_core"))
```

這讓腳本可以直接從 repo root 執行，不需要先把套件發布出去。它會把 `packages/llm_core` 加到 Python import path，讓 `llm_core` 可以被匯入。

### 2. 匯入訓練所需的 building blocks

```python
from llm_core.checkpoints import save_checkpoint
from llm_core.configs import MODEL_CONFIGS
from llm_core.model import GPTModel
from llm_core.tokenizer import ByteTokenizer
from llm_core.training import TrainingConfig, generate_sample, train_tiny_language_model
```

這些 import 對應到幾個學習元件：

- `GPTModel`：tiny GPT 架構。
- `ByteTokenizer`：把 UTF-8 bytes 轉成 token IDs。
- `TrainingConfig`：訓練參數集合。
- `generate_sample`：產生可比較的訓練前/後文字。
- `train_tiny_language_model`：執行訓練迴圈。
- `save_checkpoint`：寫出完整模型狀態快照。

### 3. 解析命令列參數

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

重要參數：

| 參數 | 意義 |
| --- | --- |
| `--dataset` | tiny training loop 使用的文字檔。 |
| `--base-model-id` | 要從哪個模型 config 開始。 |
| `--output-model-id` | 訓練後 checkpoint 載入時使用的模型名稱。 |
| `--max-steps` | optimizer 更新次數。數值越高，越容易看到 overfit。 |
| `--batch-size` | 每個 training step 使用幾段文字視窗。 |
| `--block-size` | 每段訓練視窗的 input token 數量。 |
| `--learning-rate` | AdamW 的更新步幅。 |
| `--eval-every` | 每隔幾步印出一次進度。 |
| `--sample-prompt` | 用來比較訓練前/後的 prompt。 |

### 4. 建立未訓練模型

```python
cfg = MODEL_CONFIGS[args.base_model_id]
tokenizer = ByteTokenizer()
torch.manual_seed(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(cfg.to_dict()).to(device)
model.eval()
```

這時模型已經有架構與隨機權重，但還沒有從文字中學到任何東西。

`torch.manual_seed(cfg.seed)` 讓隨機初始化可以重現，方便比較不同學習實驗。

### 5. 產生 baseline sample

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

這就是 `before training` 的輸出。它讓你看到同一個 prompt 在任何 optimizer step 發生前的 baseline。

### 6. 建立 TrainingConfig

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

這讓腳本保持簡潔，也讓 API training job 能重用相同的訓練設定結構。

### 7. 讀取 tiny dataset

```python
text = Path(args.dataset).read_text(encoding="utf-8")
```

預設資料集是 `data\tiny\every-effort.txt`。它刻意重複，讓 tiny model 可以快速 overfit，也讓短時間內的學習變化比較容易被看見。

### 8. 訓練模型

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

在 `train_tiny_language_model` 裡，訓練迴圈會反覆：

1. 把文字切成 byte-token windows。
2. 要求模型預測下一個 token。
3. 計算 cross-entropy loss。
4. 執行 backpropagation。
5. 用 AdamW 更新權重。
6. 每隔 `eval_every` steps 回報進度。

loss 是主要的學習訊號。在這個重複 tiny dataset 上，它通常會往下降，但單次印出的數值仍可能有波動。

### 9. 補上 baseline metadata

```python
summary["before_sample"] = before_sample
summary["dataset_path"] = str(Path(args.dataset))
```

checkpoint 會記錄訓練前 baseline sample 與 dataset path，之後說明這個 checkpoint 時比較容易追溯。

### 10. 儲存 full checkpoint

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

在這個專案裡，checkpoint 是完整模型狀態快照。它儲存完整的 `model.state_dict()`，不是相對於 base model 的差異檔。

也就是說，只要有相容的 `GPTModel` 程式碼與 config，這個 checkpoint 就可以直接載入成訓練後的模型狀態。

### 11. 印出學習結果

```python
print("after training:")
print(ascii(summary["sample_text"]))
print(f"checkpoint_id: {checkpoint['checkpoint_id']}")
print(f"checkpoint_path: {checkpoint['path']}")
```

腳本會印出：

- 訓練前產生的文字。
- 訓練期間的 loss 進度。
- 訓練後產生的文字。
- checkpoint ID 與路徑。

這裡使用 `ascii(...)`，是因為 byte-level random output 可能包含控制字元或看起來不像正常文字的 byte sequence。先 escape 起來，terminal 會比較容易閱讀。

## 你應該學到什麼

這支腳本示範最小完整訓練迴圈：

1. 隨機初始化模型提供訓練前 baseline。
2. 訓練使用文字 token 來降低預測錯誤。
3. loss 是看得見的學習量測值。
4. tiny repeated dataset 讓 overfit 容易被觀察。
5. full checkpoint 會把訓練後權重變成可重用的模型狀態。

當這支腳本跑通後，API training job 會更容易理解，因為它非同步執行的也是同一組核心訓練元件。

## 相關檔案

- `scripts/smoke_train.py`：命令列訓練 smoke test。
- `packages/llm_core/llm_core/training.py`：dataset、loss、optimizer loop 與 sample generation。
- `packages/llm_core/llm_core/checkpoints.py`：full checkpoint 儲存/載入工具。
- `data/tiny/every-effort.txt`：tiny training dataset。
- `docs/training-loop.zh-TW.md`：API 與 checkpoint 學習閉環。
