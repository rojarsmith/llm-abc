from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "llm_core"))

from llm_core.configs import MODEL_CONFIGS  # noqa: E402
from llm_core.generation import generate, prepare_chat_prompt  # noqa: E402
from llm_core.model import GPTModel, count_parameters  # noqa: E402
from llm_core.tokenizer import ByteTokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a minimal chat generation smoke test with an untrained tiny GPT."
    )
    parser.add_argument("--message", default="Every effort moves you")
    parser.add_argument("--model-id", default="random-tiny-byte")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_id]
    tokenizer = ByteTokenizer()
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(cfg.to_dict()).to(device)
    model.eval()

    prompt = prepare_chat_prompt(args.message)
    input_ids = tokenizer.encode(prompt)
    idx = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    output = generate(
        model=model,
        idx=idx,
        max_new_tokens=args.max_new_tokens,
        context_size=cfg.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eos_id,
    )

    output_ids = output.squeeze(0).tolist()
    generated_ids = output_ids[len(input_ids) :]
    reply = tokenizer.decode(generated_ids)

    print(f"model_id: {args.model_id}")
    print(f"device: {device}")
    print(f"parameters: {count_parameters(model):,}")
    print(f"prompt_tokens: {len(input_ids)}")
    print(f"tokens_generated: {len(generated_ids)}")
    print("")
    print("prompt:")
    print(prompt)
    print("")
    print("generated continuation:")
    print(ascii(reply))
    print("")
    print("learning note:")
    print("This model is randomly initialized. It can generate tokens, but it has not learned language yet.")


if __name__ == "__main__":
    main()
