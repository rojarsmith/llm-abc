from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from apps.api.services.chat_service import ChatRequestData, ChatService


app = FastAPI(
    title="LLM ABC API",
    version="0.1.0",
    description="A minimal educational API for a tiny ChatGPT-like model.",
)

chat_service = ChatService()
executor = ThreadPoolExecutor(max_workers=1)
jobs_lock = Lock()


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


@dataclass
class ChatJob:
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    updated_at: str
    request: dict
    result: dict | None = None
    error: str | None = None


jobs: dict[str, ChatJob] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> list[dict]:
    return chat_service.list_models()


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
    with jobs_lock:
        jobs[job_id] = job

    executor.submit(_run_chat_job, job_id, request)
    return _job_to_dict(job)


@app.get("/chat/jobs/{job_id}")
def get_chat_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Chat job not found")
        return _job_to_dict(job)


def _run_chat_job(job_id: str, request: ChatRequest) -> None:
    _update_job(job_id, status="running")
    try:
        result = chat_service.generate_reply(_to_request_data(request))
        _update_job(job_id, status="succeeded", result=result)
    except Exception as exc:  # Keep job failures visible to the UI.
        _update_job(job_id, status="failed", error=str(exc))


def _update_job(
    job_id: str,
    status: Literal["queued", "running", "succeeded", "failed"],
    result: dict | None = None,
    error: str | None = None,
) -> None:
    with jobs_lock:
        job = jobs[job_id]
        job.status = status
        job.updated_at = _utc_now()
        job.result = result
        job.error = error


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
