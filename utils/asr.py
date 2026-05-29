"""Audio Speech Recognition - supports OpenAI Whisper API and local Whisper"""
import os
import tempfile
from pathlib import Path
from openai import OpenAI


def transcribe_with_whisper_api(
    audio_path: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "whisper-1",
    language: str = "zh",
) -> dict:
    """Transcribe audio using OpenAI-compatible Whisper API

    Args:
        audio_path: Path to audio file
        api_key: API key
        base_url: API base URL
        model: Whisper model name
        language: Language code (zh, en, etc.)

    Returns:
        dict with full_text and segments
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=model,
            file=f,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    result = {
        "full_text": response.text,
        "segments": [],
    }

    if hasattr(response, "segments") and response.segments:
        for seg in response.segments:
            result["segments"].append({
                "text": getattr(seg, "text", ""),
                "start": getattr(seg, "start", 0),
                "end": getattr(seg, "end", 0),
            })

    return result


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available"""
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"],
                       capture_output=True, check=True)
        return True
    except Exception:
        return False


def convert_to_standard_wav(input_path: str, output_path: str | None = None) -> str | None:
    """Convert audio to standard 16kHz/mono/16bit WAV using ffmpeg

    Returns output path, or None if failed
    """
    import subprocess

    if output_path is None:
        output_path = input_path + "_converted.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-sample_fmt", "s16",
        "-f", "wav",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        return None
    except Exception:
        return None


def get_file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)