from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import torch

from llm_core.configs import MODEL_CONFIGS, ModelConfig
from llm_core.generation import generate, prepare_chat_prompt
from llm_core.model import GPTModel, count_parameters
from llm_core.tokenizer import ByteTokenizer


@dataclass(frozen=True)
class ChatRequestData:
    message: str
    model_id: str
    max_new_tokens: int
    temperature: float
    top_k: int | None
    include_prompt: bool


@dataclass
class LoadedModel:
    model_id: str
    config: ModelConfig
    tokenizer: ByteTokenizer
    model: GPTModel
    device: torch.device


class ChatService:
    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._locks: dict[str, Lock] = {}

    def list_models(self) -> list[dict]:
        return [
            {
                "model_id": model_id,
                "description": cfg.description,
                "tokenizer": cfg.tokenizer,
                "context_length": cfg.context_length,
                "parameters": count_parameters(GPTModel(cfg.to_dict())),
                "state": "random-untrained",
            }
            for model_id, cfg in MODEL_CONFIGS.items()
        ]

    def generate_reply(self, request: ChatRequestData) -> dict:
        loaded = self._get_model(request.model_id)
        prompt = prepare_chat_prompt(request.message)
        input_ids = loaded.tokenizer.encode(prompt)
        if not input_ids:
            input_ids = [loaded.tokenizer.eos_id]

        idx = torch.tensor(input_ids, dtype=torch.long, device=loaded.device).unsqueeze(0)

        with self._locks[request.model_id]:
            output = generate(
                model=loaded.model,
                idx=idx,
                max_new_tokens=request.max_new_tokens,
                context_size=loaded.config.context_length,
                temperature=request.temperature,
                top_k=request.top_k,
                eos_id=loaded.tokenizer.eos_id,
            )

        output_ids = output.squeeze(0).tolist()
        generated_ids = output_ids[len(input_ids) :]
        reply = loaded.tokenizer.decode(generated_ids).strip()
        full_text = loaded.tokenizer.decode(output_ids)

        return {
            "model_id": request.model_id,
            "prompt": prompt,
            "reply": full_text if request.include_prompt else reply,
            "full_text": full_text,
            "prompt_tokens": len(input_ids),
            "tokens_generated": len(generated_ids),
        }

    def _get_model(self, model_id: str) -> LoadedModel:
        if model_id not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model_id: {model_id}")

        if model_id not in self._models:
            self._models[model_id] = self._load_random_model(model_id, MODEL_CONFIGS[model_id])
            self._locks[model_id] = Lock()

        return self._models[model_id]

    def _load_random_model(self, model_id: str, config: ModelConfig) -> LoadedModel:
        if config.tokenizer != "byte":
            raise ValueError(f"Unsupported tokenizer for this MVP: {config.tokenizer}")

        torch.manual_seed(config.seed)
        tokenizer = ByteTokenizer()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GPTModel(config.to_dict()).to(device)
        model.eval()
        return LoadedModel(
            model_id=model_id,
            config=config,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )
