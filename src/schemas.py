"""Pydantic models for API request/response validation."""

from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    """Response from /analyze endpoint."""

    response: str
    model: str
    total_time_sec: float


class TranscribeResponse(BaseModel):
    """Response from /transcribe endpoint."""

    text: str
    language: str
    audio_duration_sec: float
    transcription_time_sec: float


class TTSRequest(BaseModel):
    """Request for /speak endpoint."""

    text: str


class PipelineStage(BaseModel):
    """Result from a single pipeline stage."""

    name: str
    status: str
    duration_sec: float
    within_budget: bool


class PipelineResponse(BaseModel):
    """Response from /pipeline endpoint."""

    transcript: str
    llm_response: str
    audio_path: str
    stages: list[PipelineStage]
    total_time_sec: float


class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: str
    ollama: str
    whisper: str
    piper: str
