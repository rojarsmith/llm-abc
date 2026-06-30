from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from apps.api.services.chat_service import ChatRequestData, ChatService
from apps.api.services.training_service import TrainingRequestData, TrainingService


app = FastAPI(
    title="LLM ABC API",
    version="0.1.0",
    description="A minimal educational API for a tiny ChatGPT-like model.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_service = ChatService()
training_service = TrainingService(chat_service)
executor = ThreadPoolExecutor(max_workers=1)
chat_jobs_lock = Lock()
training_jobs_lock = Lock()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    model_id: str = "random-tiny-byte"
    max_new_tokens: int = Field(32, ge=1, le=200)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_k: int | None = Field(None, ge=1, le=200)
    include_prompt: bool = False


class ChatResponse(BaseModel):
    model_id: str
    prompt: str
    reply: str
    full_text: str
    prompt_tokens: int
    tokens_generated: int


class TrainingRequest(BaseModel):
    dataset_id: str = "every-effort"
    base_model_id: str = "random-tiny-byte"
    output_model_id: str = "trained-tiny-byte"
    max_steps: int = Field(80, ge=1, le=2_000)
    batch_size: int = Field(4, ge=1, le=64)
    block_size: int = Field(32, ge=2, le=64)
    learning_rate: float = Field(3e-3, gt=0.0, le=1.0)
    eval_every: int = Field(10, ge=1, le=500)
    sample_prompt: str = Field("Every effort moves you", min_length=1)
    load_when_complete: bool = True


class ModelLoadRequest(BaseModel):
    checkpoint_id: str
    model_id: str | None = None


@dataclass
class ChatJob:
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    updated_at: str
    request: dict
    result: dict | None = None
    error: str | None = None


@dataclass
class TrainingJob:
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    updated_at: str
    request: dict
    progress: list[dict]
    result: dict | None = None
    error: str | None = None


chat_jobs: dict[str, ChatJob] = {}
training_jobs: dict[str, TrainingJob] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> list[dict]:
    return chat_service.list_models()


@app.post("/models/load")
def load_model(request: ModelLoadRequest) -> dict:
    try:
        return chat_service.load_checkpoint_model(
            checkpoint_id=request.checkpoint_id,
            model_id=request.model_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/checkpoints")
def list_checkpoints() -> list[dict]:
    return chat_service.list_checkpoints()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = chat_service.generate_reply(_to_request_data(request))
    return ChatResponse(**result)


@app.post("/chat/jobs")
def create_chat_job(request: ChatRequest) -> dict:
    job_id = str(uuid4())
    now = _utc_now()
    job = ChatJob(
        job_id=job_id,
        status="queued",
        created_at=now,
        updated_at=now,
        request=request.model_dump(),
    )
    with chat_jobs_lock:
        chat_jobs[job_id] = job

    executor.submit(_run_chat_job, job_id, request)
    return _job_to_dict(job)


@app.get("/chat/jobs/{job_id}")
def get_chat_job(job_id: str) -> dict:
    with chat_jobs_lock:
        job = chat_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Chat job not found")
        return _job_to_dict(job)


@app.get("/training/datasets")
def list_training_datasets() -> list[dict]:
    return training_service.list_datasets()


@app.post("/training/jobs")
def create_training_job(request: TrainingRequest) -> dict:
    job_id = str(uuid4())
    now = _utc_now()
    job = TrainingJob(
        job_id=job_id,
        status="queued",
        created_at=now,
        updated_at=now,
        request=request.model_dump(),
        progress=[],
    )
    with training_jobs_lock:
        training_jobs[job_id] = job

    executor.submit(_run_training_job, job_id, request)
    return _job_to_dict(job)


@app.get("/training/jobs/{job_id}")
def get_training_job(job_id: str) -> dict:
    with training_jobs_lock:
        job = training_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Training job not found")
        return _job_to_dict(job)


def _run_chat_job(job_id: str, request: ChatRequest) -> None:
    _update_chat_job(job_id, status="running")
    try:
        result = chat_service.generate_reply(_to_request_data(request))
        _update_chat_job(job_id, status="succeeded", result=result)
    except Exception as exc:  # Keep job failures visible to the UI.
        _update_chat_job(job_id, status="failed", error=str(exc))


def _run_training_job(job_id: str, request: TrainingRequest) -> None:
    _update_training_job(job_id, status="running")
    try:
        result = training_service.train(
            _to_training_request_data(request),
            progress_callback=lambda event: _append_training_progress(job_id, event),
        )
        _update_training_job(job_id, status="succeeded", result=result)
    except Exception as exc:
        _update_training_job(job_id, status="failed", error=str(exc))


def _update_chat_job(
    job_id: str,
    status: Literal["queued", "running", "succeeded", "failed"],
    result: dict | None = None,
    error: str | None = None,
) -> None:
    with chat_jobs_lock:
        job = chat_jobs[job_id]
        job.status = status
        job.updated_at = _utc_now()
        job.result = result
        job.error = error


def _update_training_job(
    job_id: str,
    status: Literal["queued", "running", "succeeded", "failed"],
    result: dict | None = None,
    error: str | None = None,
) -> None:
    with training_jobs_lock:
        job = training_jobs[job_id]
        job.status = status
        job.updated_at = _utc_now()
        job.result = result
        job.error = error


def _append_training_progress(job_id: str, event: dict) -> None:
    with training_jobs_lock:
        job = training_jobs[job_id]
        job.progress.append(event)
        job.updated_at = _utc_now()


def _job_to_dict(job: ChatJob) -> dict:
    return asdict(job)


def _to_request_data(request: ChatRequest) -> ChatRequestData:
    return ChatRequestData(
        message=request.message,
        model_id=request.model_id,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        include_prompt=request.include_prompt,
    )


def _to_training_request_data(request: TrainingRequest) -> TrainingRequestData:
    return TrainingRequestData(
        dataset_id=request.dataset_id,
        base_model_id=request.base_model_id,
        output_model_id=request.output_model_id,
        max_steps=request.max_steps,
        batch_size=request.batch_size,
        block_size=request.block_size,
        learning_rate=request.learning_rate,
        eval_every=request.eval_every,
        sample_prompt=request.sample_prompt,
        load_when_complete=request.load_when_complete,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
