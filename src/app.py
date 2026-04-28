"""
Multimodal LLM API
===================
FastAPI application with endpoints for:
  - /analyze      — upload image + text prompt → get text analysis
  - /transcribe   — upload audio → get text transcription
  - /speak        — send text → get WAV audio back
  - /pipeline     — full pipeline: audio + image → text + audio response
  - /health       — health check for all components

Usage:
  uv run uvicorn src.app:app --port 8000
"""

import logging
import os
import tempfile

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

from src.schemas import (
    AnalyzeResponse,
    HealthResponse,
    PipelineResponse,
    PipelineStage,
    TranscribeResponse,
    TTSRequest,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal LLM API",
    description="Voice + Image + Text pipeline with latency budgets",
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if all components are available."""
    # Check Ollama
    ollama_status = "unavailable"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                ollama_status = "healthy"
    except Exception:
        pass

    # Check Whisper (just verify import works)
    whisper_status = "unavailable"
    try:
        import whisper  # noqa: F401
        whisper_status = "healthy"
    except ImportError:
        pass

    # Check Piper voice model exists
    piper_status = "unavailable"
    from pathlib import Path
    model_path = Path(__file__).parent.parent / "models" / "en_US-lessac-medium.onnx"
    if model_path.exists():
        piper_status = "healthy"

    overall = "healthy" if all(
        s == "healthy" for s in [ollama_status, whisper_status, piper_status]
    ) else "degraded"

    return HealthResponse(
        status=overall,
        ollama=ollama_status,
        whisper=whisper_status,
        piper=piper_status,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image_endpoint(
    image: UploadFile = File(...),
    prompt: str = Form(default="Describe this image in detail."),
    model: str = Form(default="llava:7b"),
):
    """Upload an image and get a text analysis from LLaVA."""
    from src.vision import analyze_image

    # Save uploaded file to temp location
    suffix = os.path.splitext(image.filename or "image.png")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = analyze_image(tmp_path, prompt, model)
        return AnalyzeResponse(**result)
    finally:
        os.unlink(tmp_path)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio_endpoint(
    audio: UploadFile = File(...),
):
    """Upload an audio file and get text transcription via Whisper."""
    from src.stt import transcribe_audio

    suffix = os.path.splitext(audio.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = transcribe_audio(tmp_path)
        return TranscribeResponse(**result)
    finally:
        os.unlink(tmp_path)


@app.post("/speak")
async def speak_endpoint(request: TTSRequest):
    """Convert text to speech and return WAV audio file."""
    from src.tts import synthesize_speech

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name

    result = synthesize_speech(request.text, tmp_path)

    return FileResponse(
        tmp_path,
        media_type="audio/wav",
        filename="speech.wav",
        headers={
            "X-Synthesis-Time": str(result["synthesis_time_sec"]),
            "X-Audio-Duration": str(result["duration_sec"]),
        },
    )


@app.post("/pipeline")
async def pipeline_endpoint(
    audio: UploadFile = File(default=None),
    image: UploadFile = File(default=None),
    prompt: str = Form(default=None),
):
    """
    Full multimodal pipeline: audio + image → text + audio response.

    Accepts any combination:
      - audio only → STT → LLM → TTS
      - image only → LLM (vision) → TTS
      - audio + image → STT → LLM (vision) → TTS
      - text prompt + image → LLM (vision) → TTS
    """
    from src.pipeline import run_pipeline

    # Save uploaded files to temp locations
    audio_path = None
    image_path = None

    try:
        if audio and audio.filename:
            suffix = os.path.splitext(audio.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await audio.read())
                audio_path = tmp.name

        if image and image.filename:
            suffix = os.path.splitext(image.filename)[1] or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await image.read())
                image_path = tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            output_audio_path = tmp.name

        result = run_pipeline(
            audio_path=audio_path,
            text_prompt=prompt,
            image_path=image_path,
            output_audio_path=output_audio_path,
        )

        # Build response
        breakdown = result.latency_breakdown()
        stages = []
        for stage_name in ["stt", "llm", "tts"]:
            if stage_name in breakdown:
                info = breakdown[stage_name]
                stages.append(PipelineStage(
                    name=stage_name,
                    status=info["status"],
                    duration_sec=info["duration_sec"],
                    within_budget=info.get("within_budget", True),
                ))

        return PipelineResponse(
            transcript=result.transcript,
            llm_response=result.llm_response,
            audio_path=output_audio_path if result.audio_path else "",
            stages=stages,
            total_time_sec=result.total_time_sec,
        )

    finally:
        if audio_path:
            os.unlink(audio_path)
        if image_path:
            os.unlink(image_path)


@app.get("/pipeline/latency")
async def latency_budget():
    """Show the latency budget configuration for each stage."""
    from src.pipeline import LATENCY_BUDGET

    return {
        "stages": LATENCY_BUDGET,
        "total_budget_sec": sum(LATENCY_BUDGET.values()),
        "note": "Each stage has a max time. If exceeded, a warning is logged.",
    }
