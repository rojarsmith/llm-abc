from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from apps.api.services.chat_service import ChatService
from llm_core.checkpoints import save_checkpoint
from llm_core.configs import MODEL_CONFIGS
from llm_core.model import GPTModel
from llm_core.tokenizer import ByteTokenizer
from llm_core.training import TrainingConfig, generate_sample, train_tiny_language_model


@dataclass(frozen=True)
class TrainingRequestData:
    dataset_id: str
    base_model_id: str
    output_model_id: str
    max_steps: int
    batch_size: int
    block_size: int
    learning_rate: float
    eval_every: int
    sample_prompt: str
    load_when_complete: bool


class TrainingService:
    def __init__(self, chat_service: ChatService, project_root: Path | None = None) -> None:
        self._chat_service = chat_service
        self._project_root = project_root or Path(__file__).resolve().parents[3]
        self._checkpoint_dir = self._project_root / "models" / "checkpoints"
        self._datasets = {
            "every-effort": self._project_root / "data" / "tiny" / "every-effort.txt"
        }

    def list_datasets(self) -> list[dict]:
        return [
            {
                "dataset_id": dataset_id,
                "path": str(path),
                "exists": path.exists(),
            }
            for dataset_id, path in self._datasets.items()
        ]

    def train(
        self,
        request: TrainingRequestData,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        if request.base_model_id not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported base_model_id: {request.base_model_id}")
        dataset_path = self._datasets.get(request.dataset_id)
        if dataset_path is None:
            raise ValueError(f"Unknown dataset_id: {request.dataset_id}")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        model_config = MODEL_CONFIGS[request.base_model_id]
        tokenizer = ByteTokenizer()
        torch.manual_seed(model_config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GPTModel(model_config.to_dict()).to(device)
        model.eval()

        before_sample = generate_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=request.sample_prompt,
            device=device,
            context_size=model_config.context_length,
            max_new_tokens=24,
        )
        training_config = TrainingConfig(
            max_steps=request.max_steps,
            batch_size=request.batch_size,
            block_size=request.block_size,
            learning_rate=request.learning_rate,
            eval_every=request.eval_every,
            sample_prompt=request.sample_prompt,
            seed=model_config.seed,
        )
        training_summary = train_tiny_language_model(
            model=model,
            tokenizer=tokenizer,
            text=dataset_path.read_text(encoding="utf-8"),
            device=device,
            config=training_config,
            progress_callback=progress_callback,
        )
        training_summary["before_sample"] = before_sample
        training_summary["dataset_id"] = request.dataset_id
        training_summary["dataset_path"] = str(dataset_path)

        checkpoint = save_checkpoint(
            checkpoint_dir=self._checkpoint_dir,
            model=model,
            model_id=request.output_model_id,
            base_model_id=request.base_model_id,
            model_config=model_config,
            tokenizer_name="byte",
            training_summary=training_summary,
        )

        loaded = None
        if request.load_when_complete:
            loaded = self._chat_service.load_checkpoint_model(
                checkpoint_id=checkpoint["checkpoint_id"],
                model_id=request.output_model_id,
            )

        return {
            "checkpoint": checkpoint,
            "loaded_model": loaded,
            "training_summary": training_summary,
        }
