from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from llm_core.generation import generate, prepare_chat_prompt
from llm_core.tokenizer import ByteTokenizer


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int = 80
    batch_size: int = 4
    block_size: int = 32
    stride: int = 1
    learning_rate: float = 3e-3
    eval_every: int = 10
    sample_prompt: str = "Every effort moves you"
    sample_tokens: int = 24
    seed: int = 123


class ByteTokenDataset(Dataset):
    def __init__(self, text: str, tokenizer: ByteTokenizer, block_size: int, stride: int = 1) -> None:
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        token_ids = tokenizer.encode(text)
        if len(token_ids) <= block_size:
            raise ValueError(
                f"Training text is too short for block_size={block_size}. "
                f"Need more than {block_size} byte tokens."
            )

        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []
        for i in range(0, len(token_ids) - block_size, stride):
            input_chunk = token_ids[i : i + block_size]
            target_chunk = token_ids[i + 1 : i + block_size + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


def train_tiny_language_model(
    *,
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    text: str,
    device: torch.device,
    config: TrainingConfig,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    torch.manual_seed(config.seed)
    model.to(device)
    model.train()

    dataset = ByteTokenDataset(
        text=text,
        tokenizer=tokenizer,
        block_size=config.block_size,
        stride=config.stride,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses: list[dict] = []
    tokens_seen = 0
    step = 0

    while step < config.max_steps:
        for input_batch, target_batch in loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_batch.flatten(),
            )
            loss.backward()
            optimizer.step()

            step += 1
            tokens_seen += input_batch.numel()

            if step == 1 or step % config.eval_every == 0 or step == config.max_steps:
                event = {
                    "step": step,
                    "max_steps": config.max_steps,
                    "loss": round(float(loss.item()), 6),
                    "tokens_seen": tokens_seen,
                }
                losses.append(event)
                if progress_callback is not None:
                    progress_callback(event)

            if step >= config.max_steps:
                break

    model.eval()
    sample_text = generate_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=config.sample_prompt,
        device=device,
        context_size=model.pos_emb.num_embeddings,
        max_new_tokens=config.sample_tokens,
    )
    summary = {
        "max_steps": config.max_steps,
        "batch_size": config.batch_size,
        "block_size": config.block_size,
        "learning_rate": config.learning_rate,
        "tokens_seen": tokens_seen,
        "losses": losses,
        "final_loss": losses[-1]["loss"] if losses else None,
        "sample_prompt": config.sample_prompt,
        "sample_text": sample_text,
        "dataset_tokens": len(tokenizer.encode(text)),
    }
    return summary


def generate_sample(
    *,
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    prompt: str,
    device: torch.device,
    context_size: int,
    max_new_tokens: int,
) -> str:
    chat_prompt = prepare_chat_prompt(prompt)
    input_ids = tokenizer.encode(chat_prompt)
    idx = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    output = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=0.0,
        top_k=None,
        eos_id=tokenizer.eos_id,
    )
    output_ids = output.squeeze(0).tolist()
    generated_ids = output_ids[len(input_ids) :]
    return tokenizer.decode(generated_ids)
