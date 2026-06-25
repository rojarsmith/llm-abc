from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    description: str
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool
    tokenizer: str
    seed: int = 123

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("name")
        data.pop("description")
        data.pop("tokenizer")
        data.pop("seed")
        return data


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "random-tiny-byte": ModelConfig(
        name="random-tiny-byte",
        description="A tiny randomly initialized GPT model for learning the baseline before training.",
        vocab_size=257,
        context_length=64,
        emb_dim=64,
        n_heads=4,
        n_layers=2,
        drop_rate=0.1,
        qkv_bias=False,
        tokenizer="byte",
        seed=123,
    )
}
