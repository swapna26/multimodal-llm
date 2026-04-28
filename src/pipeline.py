"""
Pipeline Orchestrator
======================
Ties together STT → LLM (Vision) → TTS into a single pipeline.

Key concepts:
  - Latency Budget: each stage has a maximum time limit (timeout)
  - Graceful Degradation: if one stage fails, the system still works partially
  - Latency Tracking: measures time spent in each stage for observability

Latency Budget:
  STT  → max 10 seconds (transcribe audio to text)
  LLM  → max 30 seconds (analyze image + generate response)
  TTS  → max 15 seconds (convert response to speech)
  Total → max 55 seconds end-to-end

Graceful Degradation examples:
  - STT fails → return error asking user to type instead
  - LLM fails → return error "could not analyze image"
  - TTS fails → return text response without audio (still useful)

Usage:
    from src.pipeline import run_pipeline
    result = run_pipeline(audio_path="question.wav", image_path="photo.png")
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Latency budget — max time per stage (seconds)
LATENCY_BUDGET = {
    "stt": 10.0,
    "llm": 30.0,
    "tts": 15.0,
}


@dataclass
class StageResult:
    """Result from a single pipeline stage."""

    name: str
    status: str  # "success", "error", "skipped"
    duration_sec: float = 0.0
    data: dict = field(default_factory=dict)
    error: str = ""


@dataclass
class PipelineResult:
    """Result from the full pipeline run."""

    stages: list[StageResult] = field(default_factory=list)
    total_time_sec: float = 0.0

    @property
    def transcript(self) -> str:
        """Text from STT stage."""
        for s in self.stages:
            if s.name == "stt" and s.status == "success":
                return s.data.get("text", "")
        return ""

    @property
    def llm_response(self) -> str:
        """Text from LLM stage."""
        for s in self.stages:
            if s.name == "llm" and s.status == "success":
                return s.data.get("response", "")
        return ""

    @property
    def audio_path(self) -> str:
        """Audio file path from TTS stage."""
        for s in self.stages:
            if s.name == "tts" and s.status == "success":
                return s.data.get("output_path", "")
        return ""

    def latency_breakdown(self) -> dict:
        """Return timing for each stage."""
        breakdown = {}
        for s in self.stages:
            breakdown[s.name] = {
                "status": s.status,
                "duration_sec": s.duration_sec,
                "budget_sec": LATENCY_BUDGET.get(s.name, 0),
                "within_budget": s.duration_sec <= LATENCY_BUDGET.get(s.name, 999),
            }
        breakdown["total"] = {
            "duration_sec": self.total_time_sec,
            "budget_sec": sum(LATENCY_BUDGET.values()),
        }
        return breakdown


def run_pipeline(
    audio_path: str | None = None,
    text_prompt: str | None = None,
    image_path: str | None = None,
    output_audio_path: str = "output.wav",
) -> PipelineResult:
    """
    Run the full multimodal pipeline: STT → LLM → TTS.

    Args:
        audio_path: Path to audio file (optional — if provided, STT runs first)
        text_prompt: Text prompt (used if no audio, or as fallback if STT fails)
        image_path: Path to image file (optional — enables vision analysis)
        output_audio_path: Where to save the TTS output audio

    Returns:
        PipelineResult with results from each stage
    """
    result = PipelineResult()
    pipeline_start = time.perf_counter()

    # ── Stage 1: STT (Speech-to-Text) ──
    prompt = text_prompt or ""

    if audio_path:
        stage = _run_stt(audio_path)
        result.stages.append(stage)

        if stage.status == "success":
            prompt = stage.data["text"]
        elif text_prompt:
            # Graceful degradation: STT failed, fall back to text prompt
            logger.warning("STT failed, falling back to text prompt")
            prompt = text_prompt
        else:
            # No fallback available
            result.total_time_sec = time.perf_counter() - pipeline_start
            return result
    else:
        result.stages.append(StageResult(name="stt", status="skipped"))

    if not prompt:
        prompt = "Describe this image in detail."

    # ── Stage 2: LLM (Vision / Text) ──
    stage = _run_llm(prompt, image_path)
    result.stages.append(stage)

    if stage.status != "success":
        result.total_time_sec = time.perf_counter() - pipeline_start
        return result

    response_text = stage.data.get("response", "")

    # ── Stage 3: TTS (Text-to-Speech) ──
    if response_text:
        stage = _run_tts(response_text, output_audio_path)
        result.stages.append(stage)
    else:
        result.stages.append(StageResult(name="tts", status="skipped"))

    result.total_time_sec = round(time.perf_counter() - pipeline_start, 4)
    return result


def _run_stt(audio_path: str) -> StageResult:
    """Run STT stage with timeout handling."""
    start = time.perf_counter()
    try:
        from src.stt import transcribe_audio

        data = transcribe_audio(audio_path)
        elapsed = time.perf_counter() - start

        if elapsed > LATENCY_BUDGET["stt"]:
            logger.warning(f"STT exceeded budget: {elapsed:.2f}s > {LATENCY_BUDGET['stt']}s")

        return StageResult(name="stt", status="success", duration_sec=round(elapsed, 4), data=data)

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"STT failed: {e}")
        return StageResult(name="stt", status="error", duration_sec=round(elapsed, 4), error=str(e))


def _run_llm(prompt: str, image_path: str | None) -> StageResult:
    """Run LLM stage with timeout handling."""
    start = time.perf_counter()
    try:
        from src.vision import analyze_image

        if image_path:
            data = analyze_image(image_path, prompt)
        else:
            # Text-only: use Ollama directly without image
            import httpx

            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3.2:1b", "prompt": prompt, "stream": False},
                )
                resp_data = resp.json()
                data = {
                    "response": resp_data.get("response", ""),
                    "model": "llama3.2:1b",
                    "total_time_sec": round(time.perf_counter() - start, 4),
                }

        elapsed = time.perf_counter() - start

        if elapsed > LATENCY_BUDGET["llm"]:
            logger.warning(f"LLM exceeded budget: {elapsed:.2f}s > {LATENCY_BUDGET['llm']}s")

        return StageResult(name="llm", status="success", duration_sec=round(elapsed, 4), data=data)

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"LLM failed: {e}")
        return StageResult(name="llm", status="error", duration_sec=round(elapsed, 4), error=str(e))


def _run_tts(text: str, output_path: str) -> StageResult:
    """Run TTS stage with timeout handling."""
    start = time.perf_counter()
    try:
        from src.tts import synthesize_speech

        # Truncate very long responses to keep TTS fast
        if len(text) > 500:
            text = text[:500] + "..."

        data = synthesize_speech(text, output_path)
        elapsed = time.perf_counter() - start

        if elapsed > LATENCY_BUDGET["tts"]:
            logger.warning(f"TTS exceeded budget: {elapsed:.2f}s > {LATENCY_BUDGET['tts']}s")

        return StageResult(name="tts", status="success", duration_sec=round(elapsed, 4), data=data)

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"TTS failed: {e}")
        return StageResult(name="tts", status="error", duration_sec=round(elapsed, 4), error=str(e))
