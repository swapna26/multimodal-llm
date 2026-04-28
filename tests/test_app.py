"""Tests for the multimodal LLM API."""

from src.schemas import AnalyzeResponse, HealthResponse, PipelineStage, TranscribeResponse
from src.pipeline import StageResult, PipelineResult, LATENCY_BUDGET


def test_health_response_model():
    """Health response has correct fields."""
    resp = HealthResponse(
        status="healthy",
        ollama="healthy",
        whisper="healthy",
        piper="healthy",
    )
    assert resp.status == "healthy"
    assert resp.ollama == "healthy"


def test_health_response_degraded():
    """Health response reports degraded when a component is down."""
    resp = HealthResponse(
        status="degraded",
        ollama="unavailable",
        whisper="healthy",
        piper="healthy",
    )
    assert resp.status == "degraded"
    assert resp.ollama == "unavailable"


def test_analyze_response_model():
    """Analyze response has correct fields."""
    resp = AnalyzeResponse(
        response="I see a red circle",
        model="llava:7b",
        total_time_sec=2.5,
    )
    assert resp.response == "I see a red circle"
    assert resp.model == "llava:7b"


def test_transcribe_response_model():
    """Transcribe response has correct fields."""
    resp = TranscribeResponse(
        text="Hello world",
        language="en",
        audio_duration_sec=2.0,
        transcription_time_sec=0.5,
    )
    assert resp.text == "Hello world"
    assert resp.language == "en"


def test_pipeline_result_properties():
    """PipelineResult correctly extracts data from stages."""
    result = PipelineResult(
        stages=[
            StageResult(name="stt", status="success", duration_sec=1.0, data={"text": "What is this?"}),
            StageResult(name="llm", status="success", duration_sec=3.0, data={"response": "A red circle"}),
            StageResult(name="tts", status="success", duration_sec=0.5, data={"output_path": "out.wav"}),
        ],
        total_time_sec=4.5,
    )
    assert result.transcript == "What is this?"
    assert result.llm_response == "A red circle"
    assert result.audio_path == "out.wav"


def test_pipeline_graceful_degradation():
    """Pipeline handles STT failure gracefully."""
    result = PipelineResult(
        stages=[
            StageResult(name="stt", status="error", duration_sec=0.1, error="File not found"),
            StageResult(name="llm", status="success", duration_sec=2.0, data={"response": "Hello"}),
            StageResult(name="tts", status="success", duration_sec=0.3, data={"output_path": "out.wav"}),
        ],
        total_time_sec=2.4,
    )
    assert result.transcript == ""  # STT failed, no transcript
    assert result.llm_response == "Hello"  # LLM still worked


def test_latency_breakdown():
    """Latency breakdown correctly flags over-budget stages."""
    result = PipelineResult(
        stages=[
            StageResult(name="stt", status="success", duration_sec=1.0),
            StageResult(name="llm", status="success", duration_sec=35.0),  # Over budget (30s)
            StageResult(name="tts", status="success", duration_sec=2.0),
        ],
        total_time_sec=38.0,
    )
    breakdown = result.latency_breakdown()
    assert breakdown["stt"]["within_budget"] is True
    assert breakdown["llm"]["within_budget"] is False  # 35s > 30s budget
    assert breakdown["tts"]["within_budget"] is True


def test_latency_budget_values():
    """Latency budget has expected stages."""
    assert "stt" in LATENCY_BUDGET
    assert "llm" in LATENCY_BUDGET
    assert "tts" in LATENCY_BUDGET
    assert LATENCY_BUDGET["stt"] == 10.0
    assert LATENCY_BUDGET["llm"] == 30.0


def test_pipeline_stage_schema():
    """PipelineStage pydantic model works correctly."""
    stage = PipelineStage(
        name="stt",
        status="success",
        duration_sec=1.5,
        within_budget=True,
    )
    assert stage.name == "stt"
    assert stage.within_budget is True
