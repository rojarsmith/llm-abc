from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from llm_core.configs import ModelConfig


def default_checkpoints_dir(project_root: Path | str | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path.cwd()
    return root / "models" / "checkpoints"


def save_checkpoint(
    *,
    checkpoint_dir: Path,
    model: torch.nn.Module,
    model_id: str,
    base_model_id: str,
    model_config: ModelConfig,
    tokenizer_name: str,
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    checkpoint_id = _checkpoint_id(model_id, created_at)
    checkpoint_path = checkpoint_dir / f"{checkpoint_id}.pt"

    payload = {
        "checkpoint_id": checkpoint_id,
        "model_id": model_id,
        "base_model_id": base_model_id,
        "created_at": created_at,
        "model_config": asdict(model_config),
        "tokenizer": tokenizer_name,
        "training_summary": training_summary,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_metadata(checkpoint_path, payload)


def load_checkpoint(checkpoint_path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=map_location)


def list_checkpoints(checkpoint_dir: Path) -> list[dict[str, Any]]:
    if not checkpoint_dir.exists():
        return []

    checkpoints: list[dict[str, Any]] = []
    for checkpoint_path in sorted(checkpoint_dir.glob("*.pt"), reverse=True):
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoints.append(checkpoint_metadata(checkpoint_path, payload))
    return checkpoints


def find_checkpoint(checkpoint_dir: Path, checkpoint_id: str) -> Path:
    checkpoint_path = checkpoint_dir / f"{checkpoint_id}.pt"
    if checkpoint_path.exists():
        return checkpoint_path
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")


def checkpoint_metadata(checkpoint_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    training_summary = payload.get("training_summary", {})
    return {
        "checkpoint_id": payload["checkpoint_id"],
        "model_id": payload["model_id"],
        "base_model_id": payload["base_model_id"],
        "created_at": payload["created_at"],
        "path": str(checkpoint_path),
        "tokenizer": payload.get("tokenizer", "byte"),
        "training_summary": training_summary,
    }


def _checkpoint_id(model_id: str, created_at: str) -> str:
    safe_model_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in model_id)
    timestamp = (
        created_at.replace("-", "")
        .replace(":", "")
        .replace(".", "")
        .replace("+", "z")
    )
    return f"{safe_model_id}-{timestamp}"
