# `smoke_chat.py` Explained

[English](smoke-chat.md) | [繁體中文](smoke-chat.zh-TW.md)

This document explains what `scripts/smoke_chat.py` does and why it exists.

The script is the first learning checkpoint in this project. It lets you run the model core without starting FastAPI, without building a Web UI, and without training anything first.

## Why This Script Exists

`smoke_chat.py` answers one narrow question:

Can our local GPT model code accept text, turn it into tokens, run a forward pass, and generate new tokens?

At this stage, a good result is not a smart answer. A good result is proof that the full inference path works:

```text
message -> prompt -> token ids -> GPTModel -> generated token ids -> text
```

Because the model is randomly initialized, the output should look random. That is the baseline you compare against after training.

## How to Run It

Use Windows Command Prompt (`cmd.exe`). Create and activate the project-local `.venv` first:

```cmd
where python
python --version
python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 'Use Python 3.11, 3.12, or 3.13 for this project')"

python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e . -r apps\api\requirements.txt
```

If `python --version` shows Python 3.14, adjust your Windows PATH so `python` points to Python 3.11, 3.12, or 3.13.

Then run:

```cmd
python scripts/smoke_chat.py --message "Every effort moves you" --max-new-tokens 24
```

Example output shape:

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

## If You See an `actions-runner` Warning

You may see a warning like this:

```text
UserWarning: Failed to initialize NumPy: No module named 'numpy'
... C:\actions-runner\_work\pytorch\pytorch\...
```

The `actions-runner` path does not mean GitHub Actions is running on your machine. It is a source path embedded in the prebuilt PyTorch wheel from the machine that built PyTorch.

The actionable part is `No module named 'numpy'`. Install the project dependencies again inside `.venv`:

```cmd
python -m pip install -e . -r apps\api\requirements.txt
```

The project includes `numpy` as a dependency so this warning should disappear after reinstalling dependencies.

## Step-by-Step Flow

### 1. Make the local package importable

The script adds `packages/llm_core` to Python's import path:

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "llm_core"))
```

This lets the script import `llm_core` directly even before the project is installed as a package.

### 2. Read command-line arguments

The script accepts:

```text
--message
--model-id
--max-new-tokens
--temperature
--top-k
```

These arguments let you change the prompt and generation behavior without editing code.

### 3. Load the model configuration

```python
cfg = MODEL_CONFIGS[args.model_id]
```

The default `model_id` is `random-tiny-byte`. Its config currently defines a very small GPT:

```text
vocab_size: 257
context_length: 64
emb_dim: 64
n_heads: 4
n_layers: 2
```

This is intentionally tiny so the first loop can run quickly on a local machine.

### 4. Create the tokenizer

```python
tokenizer = ByteTokenizer()
```

The first version uses `ByteTokenizer` instead of `tiktoken`.

That keeps the first learning loop dependency-light:

- UTF-8 bytes become token ids `0..255`.
- Token id `256` is reserved as EOS.
- The vocabulary size is `257`.

This tokenizer is not meant to be the final GPT-2-compatible tokenizer. It is a teaching tool for the first backend loop.

### 5. Fix the random seed

```python
torch.manual_seed(cfg.seed)
```

The model weights are still random, but the random initialization is repeatable. This makes the smoke test easier to compare across runs.

### 6. Pick CPU or GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If CUDA is available, the script uses the GPU. Otherwise it uses CPU.

### 7. Create the GPT model

```python
model = GPTModel(cfg.to_dict()).to(device)
model.eval()
```

This constructs the tiny GPT architecture from scratch. It does not load trained weights.

`model.eval()` switches the model to inference mode. This matters because dropout should be disabled during generation.

### 8. Build the chat-style prompt

```python
prompt = prepare_chat_prompt(args.message)
```

For the message:

```text
Every effort moves you
```

the prompt becomes:

```text
User: Every effort moves you
Assistant:
```

This simple format prepares the later Web Chat API shape.

### 9. Convert text into token ids

```python
input_ids = tokenizer.encode(prompt)
idx = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
```

The tokenizer converts text into integers. The tensor shape becomes:

```text
[batch_size, sequence_length]
```

For a single prompt, `batch_size` is `1`.

### 10. Generate new tokens

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

`generate()` repeatedly:

1. Crops the context to the model's context length.
2. Runs the GPT model.
3. Reads the logits for the last token.
4. Chooses the next token.
5. Appends that token to the sequence.

This is the core autoregressive language model loop.

### 11. Decode only the generated continuation

```python
output_ids = output.squeeze(0).tolist()
generated_ids = output_ids[len(input_ids) :]
reply = tokenizer.decode(generated_ids)
```

The full output contains the original prompt plus generated tokens. The script removes the prompt part so `reply` only shows what the model generated after `Assistant:`.

### 12. Print learning-oriented metadata

The script prints:

- `model_id`: which model config was used.
- `device`: CPU or CUDA.
- `parameters`: trainable parameter count.
- `prompt_tokens`: how many input tokens were created from the prompt.
- `tokens_generated`: how many new tokens were produced.
- `generated continuation`: the decoded generated text.

The generated text is printed with `ascii(reply)` because an untrained byte-level model can produce arbitrary bytes that may not display cleanly in every terminal.

## Argument Reference

| Argument | Default | Meaning |
|---|---:|---|
| `--message` | `Every effort moves you` | User message inserted into the prompt. |
| `--model-id` | `random-tiny-byte` | Model config key from `MODEL_CONFIGS`. |
| `--max-new-tokens` | `24` | Number of new tokens to generate. |
| `--temperature` | `0.0` | `0` means deterministic argmax; higher values sample more randomly. |
| `--top-k` | `None` | If set, only the top `k` logits are eligible during sampling. |

## What You Should Learn

The important lesson is the separation between capability and training.

The code has enough structure to generate tokens:

- tokenizer
- embeddings
- transformer blocks
- output head
- autoregressive generation loop

But the model has no learned language behavior yet, because the weights are random.

That is why this smoke test should be run before training. Later, when training is added, you can run the same prompt again and compare the before-and-after behavior.

## Related Files

- [`scripts/smoke_chat.py`](../scripts/smoke_chat.py)
- [`packages/llm_core/llm_core/configs.py`](../packages/llm_core/llm_core/configs.py)
- [`packages/llm_core/llm_core/tokenizer.py`](../packages/llm_core/llm_core/tokenizer.py)
- [`packages/llm_core/llm_core/model.py`](../packages/llm_core/llm_core/model.py)
- [`packages/llm_core/llm_core/generation.py`](../packages/llm_core/llm_core/generation.py)
