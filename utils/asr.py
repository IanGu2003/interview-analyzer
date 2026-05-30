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
#  Alibaba Cloud (阿里云) ASR - 录音文件识别
# ═══════════════════════════════════════════════════════════════

def _get_oss_bucket(access_key_id: str, access_key_secret: str,
                    oss_endpoint: str, oss_bucket: str):
    """Get OSS bucket client"""
    import oss2
    auth = oss2.Auth(access_key_id, access_key_secret)
    return oss2.Bucket(auth, oss_endpoint, oss_bucket)


def _upload_audio_to_oss(audio_path: str, bucket, prefix: str = "asr_temp/") -> str:
    """Upload audio to OSS and return signed URL (24h expiry)"""
    filename = os.path.basename(audio_path)
    oss_key = f"{prefix}{uuid.uuid4().hex}_{filename}"
    bucket.put_object_from_file(oss_key, audio_path)
    # Generate signed URL with 24h expiry
    url = bucket.sign_url('GET', oss_key, 24 * 3600)
    return url


def _submit_asr_task(client, app_key: str, audio_url: str,
                     language: str = "zh-CN") -> str:
    """Submit recording file recognition task, returns task_id"""
    from aliyunsdkcore.request import CommonRequest

    request = CommonRequest()
    request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    request.set_version('2019-02-28')
    request.set_action_name('SubmitTask')
    request.set_method('POST')

    task = {
        'app_key': app_key,
        'file_link': audio_url,
        'language_code': language,
    }
    request.set_body(json.dumps(task))

    response = client.do_action_with_exception(request)
    result = json.loads(response)

    if result.get('Status') != 'SUCCESS':
        raise Exception(f"阿里云ASR提交任务失败: {result.get('Status')} - {result}")

    return result['Data']['TaskId']


def _get_task_result(client, task_id: str) -> dict:
    """Poll for task result"""
    from aliyunsdkcore.request import CommonRequest

    request = CommonRequest()
    request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    request.set_version('2019-02-28')
    request.set_action_name('GetTaskResult')
    request.set_method('GET')
    request.add_query_param('TaskId', task_id)

    response = client.do_action_with_exception(request)
    return json.loads(response)


def transcribe_with_aliyun(
    audio_path: str,
    access_key_id: str,
    access_key_secret: str,
    oss_endpoint: str,
    oss_bucket: str,
    nls_app_key: str,
    region: str = "cn-shanghai",
    language: str = "zh-CN",
) -> dict:
    """Transcribe audio using Alibaba Cloud 录音文件识别

    流程：上传音频到OSS → 提交录音文件识别任务 → 轮询获取结果

    Args:
        audio_path: Path to audio file
        access_key_id: 阿里云 AccessKey ID
        access_key_secret: 阿里云 AccessKey Secret
        oss_endpoint: OSS Endpoint, e.g. "oss-cn-shanghai.aliyuncs.com"
        oss_bucket: OSS Bucket name
        nls_app_key: 智能语音交互 AppKey
        region: 阿里云 region, default "cn-shanghai"
        language: 语言代码, default "zh-CN"

    Returns:
        dict with full_text and segments
    """
    from aliyunsdkcore.client import AcsClient

    # Step 1: Upload audio to OSS
    bucket = _get_oss_bucket(access_key_id, access_key_secret, oss_endpoint, oss_bucket)
    audio_url = _upload_audio_to_oss(audio_path, bucket)

    # Step 2: Initialize AcsClient and submit task
    client = AcsClient(access_key_id, access_key_secret, region)
    task_id = _submit_asr_task(client, nls_app_key, audio_url, language)

    # Step 3: Poll for result
    max_attempts = 120  # ~6 minutes max
    for attempt in range(max_attempts):
        time.sleep(3)
        result = _get_task_result(client, task_id)
        status = result.get('Status', '')

        if status == 'SUCCESS':
            # Parse recognition result
            sentences = []
            full_text_parts = []

            for item in result.get('Result', []):
                text = item.get('Text', '')
                sentences.append({
                    "text": text,
                    "start": item.get('BeginTime', 0) / 1000,
                    "end": item.get('EndTime', 0) / 1000,
                    "channel": item.get('ChannelId', 0),
                })
                full_text_parts.append(text)

            return {
                "full_text": "".join(full_text_parts),
                "segments": sentences,
            }

        elif status == 'FAIL':
            error_msg = result.get('StatusText', '未知错误')
            raise Exception(f"阿里云ASR识别失败: {error_msg}")

        # else RUNNING - continue polling

    raise Exception("阿里云ASR识别超时（超过6分钟）")


def check_aliyun_asr_deps() -> tuple[bool, str]:
    """Check if Alibaba Cloud ASR dependencies are installed"""
    missing = []
    try:
        import oss2
    except ImportError:
        missing.append("oss2")
    try:
        from aliyunsdkcore.client import AcsClient
    except ImportError:
        missing.append("aliyun-python-sdk-core")

    if missing:
        return False, f"缺少依赖: {', '.join(missing)}。请在终端运行: pip install {' '.join(missing)}"
    return True, "依赖已就绪"