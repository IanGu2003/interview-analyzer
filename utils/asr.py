"""Audio Speech Recognition - supports OpenAI Whisper API and Alibaba Cloud ASR"""
import os
import uuid
import json
import time
import base64
import hashlib
import hmac
import tempfile
from pathlib import Path
from datetime import datetime
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


def _get_nls_token(access_key_id: str, access_key_secret: str,
                   region: str = "cn-shanghai") -> str:
    """Get NLS authorization token using SDK's built-in ROA signer"""
    import requests
    from aliyunsdkcore.auth.composer import roa_signature_composer

    host = f'nls-meta.{region}.aliyuncs.com'
    path = '/pop/2018-05-18/tokens'
    url = f'https://{host}{path}'
    method = 'POST'

    # Build request headers for ROA signing
    from datetime import datetime
    import base64, hashlib
    
    date_str = datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    body_md5 = base64.b64encode(hashlib.md5(b'').digest()).decode()
    
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Date': date_str,
        'Content-MD5': body_md5,
    }

    # Use SDK's built-in ROA signature composer (returns tuple: headers, string_to_sign)
    signed_headers, _str_to_sign = roa_signature_composer.get_signature_headers(
        queries={},
        access_key=access_key_id,
        secret=access_key_secret,
        format='JSON',
        headers=headers,
        uri_pattern=path,
        paths={},
        method=method,
    )

    # ROA composer may not include Date/Content-MD5 in signed output, re-add them
    signed_headers['Date'] = date_str
    signed_headers['Content-MD5'] = body_md5
    signed_headers['Content-Type'] = 'application/json'
    signed_headers['Accept'] = 'application/json'

    resp = requests.post(url, headers=signed_headers, data=b'', timeout=15)
    if resp.status_code != 200:
        raise Exception(
            f"获取NLS Token失败 ({resp.status_code}): {resp.text}")

    result = resp.json()
    token = result.get('Token', {}).get('Id', '')
    if not token:
        raise Exception(f"获取NLS Token失败: 返回中无Token字段: {result}")
    return token


def _submit_asr_task_rest(nls_token: str, app_key: str, audio_url: str,
                          language: str = "zh-CN") -> str:
    """Submit recording file recognition task via REST API, returns task_id"""
    import requests

    url = "https://nls-meta.cn-shanghai.aliyuncs.com/api/v1/recog/asr"
    headers = {
        "X-NLS-Token": nls_token,
        "Content-Type": "application/json",
    }
    payload = {
        "app_key": app_key,
        "file_link": audio_url,
        "language_code": language,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=30)

    if resp.status_code != 200:
        raise Exception(
            f"阿里云ASR提交任务失败 (HTTP {resp.status_code}): {resp.text[:500]}"
        )

    if not resp.text.strip():
        raise Exception(
            f"阿里云ASR提交任务失败: 响应体为空 (HTTP {resp.status_code}, "
            f"token前10位: {nls_token[:10]}..., app_key: {app_key[:6]}...)"
        )

    result = resp.json()

    if result.get("Status") == "SUCCESS":
        task_id = result.get("Data", {}).get("TaskId", "")
        if task_id:
            return task_id

    raise Exception(f"阿里云ASR提交任务失败: {result}")


def _get_task_result_rest(nls_token: str, task_id: str) -> dict:
    """Poll for task result via REST API"""
    import requests

    url = f"https://nls-meta.cn-shanghai.aliyuncs.com/api/v1/recog/asr?task_id={task_id}"
    headers = {"X-NLS-Token": nls_token}

    resp = requests.get(url, headers=headers, timeout=30)
    return resp.json()


def transcribe_with_aliyun(
    audio_path: str,
    access_key_id: str,
    access_key_secret: str,
    oss_endpoint: str,
    oss_bucket: str,
    nls_app_key: str,
    region: str = "cn-shanghai",
    language: str = "zh-CN",
    progress_callback: callable = None,
) -> dict:
    """Transcribe audio using Alibaba Cloud 录音文件识别

    流程：上传音频到OSS → 获取NLS Token → 提交任务 → 轮询获取结果

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

    # Step 1: Upload audio to OSS
    if progress_callback:
        progress_callback("上传音频到OSS...")
    bucket = _get_oss_bucket(access_key_id, access_key_secret, oss_endpoint, oss_bucket)
    audio_url = _upload_audio_to_oss(audio_path, bucket)
    print(f"  音频已上传OSS: {audio_url}")

    # Step 2: Get NLS Token
    if progress_callback:
        progress_callback("获取NLS认证Token...")
    nls_token = _get_nls_token(access_key_id, access_key_secret, region)

    # Step 3: Submit ASR task
    if progress_callback:
        progress_callback("提交录音文件识别任务...")
    task_id = _submit_asr_task_rest(nls_token, nls_app_key, audio_url, language)
    print(f"  已提交ASR任务: {task_id}")

    # Step 4: Poll for result
    max_attempts = 120
    for attempt in range(max_attempts):
        time.sleep(3)
        if progress_callback:
            progress_callback(f"等待识别结果（{attempt * 3 + 3}秒）...")
        result = _get_task_result_rest(nls_token, task_id)
        status = result.get('Status', '')

        if status == 'SUCCESS':
            # Parse recognition result
            sentences = []
            full_text_parts = []

            # Result may be a dict with Sentences, or a list
            raw_result = result.get('Result', [])
            items = []
            if isinstance(raw_result, dict):
                items = raw_result.get('Sentences', [])
            elif isinstance(raw_result, list):
                items = raw_result
            else:
                # Try StatusText
                status_text_result = result.get('StatusText', '')
                if status_text_result:
                    return {
                        "full_text": status_text_result,
                        "segments": [],
                    }

            for item in items:
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