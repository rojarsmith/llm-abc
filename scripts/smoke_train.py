from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "llm_core"))

from llm_core.checkpoints import save_checkpoint  # noqa: E402
from llm_core.configs import MODEL_CONFIGS  # noqa: E402
from llm_core.model import GPTModel  # noqa: E402
from llm_core.tokenizer import ByteTokenizer  # noqa: E402
from llm_core.training import TrainingConfig, generate_sample, train_tiny_language_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the tiny byte-level GPT on a tiny dataset and save a checkpoint."
    )
    parser.add_argument("--dataset", default=str(ROOT / "data" / "tiny" / "every-effort.txt"))
    parser.add_argument("--base-model-id", default="random-tiny-byte")
    parser.add_argument("--output-model-id", default="trained-tiny-byte")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--sample-prompt", default="Every effort moves you")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.base_model_id]
    tokenizer = ByteTokenizer()
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(cfg.to_dict()).to(device)
    model.eval()

    before_sample = generate_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=args.sample_prompt,
        device=device,
        context_size=cfg.context_length,
        max_new_tokens=24,
    )

    print(f"base_model_id: {args.base_model_id}")
    print(f"output_model_id: {args.output_model_id}")
    print(f"device: {device}")
    print("")
    print("before training:")
    print(ascii(before_sample))
    print("")
    print("training progress:")

    training_config = TrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        sample_prompt=args.sample_prompt,
        seed=cfg.seed,
    )
    text = Path(args.dataset).read_text(encoding="utf-8")
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
    summary["before_sample"] = before_sample
    summary["dataset_path"] = str(Path(args.dataset))

    checkpoint = save_checkpoint(
        checkpoint_dir=ROOT / "models" / "checkpoints",
        model=model,
        model_id=args.output_model_id,
        base_model_id=args.base_model_id,
        model_config=cfg,
        tokenizer_name="byte",
        training_summary=summary,
    )

    print("")
    print("after training:")
    print(ascii(summary["sample_text"]))
    print("")
    print(f"checkpoint_id: {checkpoint['checkpoint_id']}")
    print(f"checkpoint_path: {checkpoint['path']}")
    print("")
    print("learning note:")
    print("The tiny model is trained on a repeated tiny dataset, so it should overfit quickly.")


if __name__ == "__main__":
    main()
