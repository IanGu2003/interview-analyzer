"""Audio Speech Recognition - supports OpenAI Whisper API and Alibaba Cloud ASR"""
import os
import uuid
import json
import time
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


# ═══════════════════════════════════════════════════════════════
#  Alibaba Cloud (阿里云) ASR - 通过 DashScope（百炼）API
# ═══════════════════════════════════════════════════════════════

def transcribe_with_aliyun(
    audio_path: str,
    access_key_id: str,
    access_key_secret: str,
    oss_endpoint: str,
    oss_bucket: str,
    dashscope_api_key: str,
    progress_callback: callable = None,
) -> dict:
    """Transcribe audio using Alibaba Cloud DashScope Paraformer ASR

    流程：
    1. 上传音频到OSS
    2. 调用 DashScope ASR API（transcription-async 异步接口）

    Args:
        audio_path: Path to audio file
        access_key_id: 阿里云 AccessKey ID（用于OSS上传）
        access_key_secret: 阿里云 AccessKey Secret
        oss_endpoint: OSS Endpoint, e.g. "oss-cn-hangzhou.aliyuncs.com"
        oss_bucket: OSS Bucket name
        dashscope_api_key: DashScope API Key（百炼平台获取）
        progress_callback: Optional progress callback

    Returns:
        dict with full_text and segments
    """
    import oss2
    import requests

    # Step 1: Upload audio to OSS
    if progress_callback:
        progress_callback("上传音频到OSS...")
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, oss_endpoint, oss_bucket)
    filename = os.path.basename(audio_path)
    oss_key = f"asr_temp/{uuid.uuid4().hex}_{filename}"
    bucket.put_object_from_file(oss_key, audio_path)
    # Generate signed URL and ensure it uses HTTPS
    audio_url = bucket.sign_url('GET', oss_key, 24 * 3600)
    if audio_url.startswith("http://"):
        audio_url = "https://" + audio_url[7:]

    # Step 2: Submit async ASR task
    if progress_callback:
        progress_callback("提交DashScope异步识别任务...")

    submit_url = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription-async"
    headers = {
        "Authorization": f"Bearer {dashscope_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "paraformer-v1",
        "input": {
            "file_urls": [audio_url],
        },
    }

    resp = requests.post(submit_url, json=payload, headers=headers, timeout=30)
    if resp.status_code not in (200, 202):
        raise Exception(
            f"DashScope ASR提交失败 (HTTP {resp.status_code}): {resp.text[:500]}"
        )

    result = resp.json()
    task_id = result.get("output", {}).get("task_id")
    if not task_id:
        raise Exception(f"DashScope未返回task_id: {resp.text[:300]}")

    # Step 3: Poll for completion
    query_url = f"{submit_url}/{task_id}"
    poll_start = time.time()

    while True:
        elapsed = int(time.time() - poll_start)
        if progress_callback:
            progress_callback(f"等待识别结果 ({elapsed}秒)...")

        resp = requests.get(query_url, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"查询任务失败 (HTTP {resp.status_code}): {resp.text[:300]}")

        status_result = resp.json()
        status = status_result.get("output", {}).get("task_status", "")

        if status == "SUCCEEDED":
            break
        elif status == "FAILED":
            msg = status_result.get("output", {}).get("message", "未知错误")
            raise Exception(f"DashScope ASR识别失败: {msg}")
        elif status in ("PENDING", "RUNNING"):
            if elapsed > 600:
                raise Exception("DashScope ASR超时（超过10分钟）")
            # Adaptive polling interval
            sleep_time = 5 if elapsed < 60 else 10
            time.sleep(sleep_time)
            continue
        else:
            raise Exception(f"DashScope未知状态: {status}")

    # Step 4: Get result
    if progress_callback:
        progress_callback("获取识别结果...")

    result_url = f"{query_url}/result"
    resp = requests.get(result_url, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"获取识别结果失败 (HTTP {resp.status_code}): {resp.text[:300]}")

    result_data = resp.json()
    output = result_data.get("output", {})
    transcription = output.get("results", [{}])[0] if output.get("results") else {}
    full_text = transcription.get("text", "")
    sentences_raw = transcription.get("sentences", [])

    segments = []
    for sent in sentences_raw:
        segments.append({
            "text": sent.get("text", ""),
            "start": sent.get("begin_time", 0) / 1000,
            "end": sent.get("end_time", 0) / 1000,
            "channel": sent.get("channel_id", 0),
        })

    return {
        "full_text": full_text,
        "segments": segments,
    }


def check_aliyun_asr_deps() -> tuple[bool, str]:
    """Check if Alibaba Cloud ASR dependencies are installed"""
    missing = []
    try:
        import oss2
    except ImportError:
        missing.append("oss2")
    try:
        import requests
    except ImportError:
        missing.append("requests")

    if missing:
        return False, f"缺少依赖: {', '.join(missing)}"
    return True, "依赖已就绪"