"""
音频转写工具 - 将音频文件转换为文字
- 支持本地音频文件和URL
- 自动使用ffmpeg转码为标准格式（16kHz, 16bit, mono WAV）
- 解决服务端audio convert failed问题（错误码11103/21109）
"""
import base64
import os
import json
import subprocess
import tempfile
import uuid
import time
from typing import Dict, Any
from langchain.tools import tool
from coze_coding_dev_sdk import ASRClient
from coze_coding_utils.runtime_ctx.context import new_context
import requests


# 标准ASR音频参数
TARGET_SAMPLE_RATE = "16000"   # 16kHz
TARGET_CHANNELS = "1"          # mono
TARGET_FORMAT = "wav"          # WAV格式
TARGET_BIT_DEPTH = "16"        # 16bit


def check_ffmpeg() -> bool:
    """检查ffmpeg是否可用"""
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception as e:
        print(f"[DEBUG] check_ffmpeg failed: {e}")
        return False


def get_file_size_mb(path: str) -> float:
    """获取文件大小（MB）"""
    return os.path.getsize(path) / (1024 * 1024)


def convert_audio_with_ffmpeg(input_path: str, output_path: str) -> tuple[bool, str]:
    """
    使用ffmpeg将音频转换为标准16kHz/mono/16bit WAV

    Returns:
        (成功标志, 详细信息)
    """
    try:
        # 先探测原始音频信息
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            input_path
        ]
        probe_r = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        probe_info = ""
        if probe_r.returncode == 0:
            info = json.loads(probe_r.stdout)
            streams = info.get("streams", [])
            if streams:
                s = streams[0]
                probe_info = f"codec={s.get('codec_name')}, sr={s.get('sample_rate')}, ch={s.get('channels')}, dur={s.get('duration', '?')}s"
                print(f"[DEBUG] 原始音频信息: {probe_info}")

        # 执行转码
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", TARGET_SAMPLE_RATE,
            "-ac", TARGET_CHANNELS,
            "-sample_fmt", "s16",
            "-f", "wav",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        if result.returncode != 0:
            err = result.stderr[-300:] if result.stderr else "无错误输出"
            return False, f"ffmpeg返回码{result.returncode}: {err}"

        if not os.path.exists(output_path):
            return False, "转码后文件不存在"

        out_size = os.path.getsize(output_path)
        if out_size == 0:
            return False, "转码后文件为空"

        return True, f"转码成功: {out_size/1024/1024:.1f}MB, {probe_info}"

    except subprocess.TimeoutExpired:
        return False, "ffmpeg转码超时(>180s)"
    except Exception as e:
        return False, f"ffmpeg异常: {e}"


def download_file(url: str, local_path: str, max_size_mb: int = 100) -> tuple[bool, str]:
    """从URL下载文件到本地"""
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()

        # 检查Content-Length
        content_length = resp.headers.get("Content-Length")
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > max_size_mb:
                return False, f"文件大小{size_mb:.1f}MB超过{max_size_mb}MB限制"

        downloaded = 0
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded > max_size_mb * 1024 * 1024:
                    os.remove(local_path)
                    return False, f"下载超过{max_size_mb}MB限制"

        actual_mb = downloaded / (1024 * 1024)
        return True, f"下载完成: {actual_mb:.1f}MB"

    except requests.exceptions.RequestException as e:
        return False, f"下载请求失败: {e}"
    except Exception as e:
        return False, f"下载异常: {e}"


@tool
def speech_to_text(audio_path: str, uid: str = "interview_user") -> str:
    """
    将音频文件转换为文字（支持 .mp3, .wav, .ogg, .m4a 等格式）
    支持本地路径和HTTP/HTTPS URL
    自动使用ffmpeg将音频转为标准16kHz/mono/16bit WAV再提交

    Args:
        audio_path: 音频文件路径或URL
        uid: 用户唯一标识符

    Returns:
        包含转写文字和时间戳信息的JSON字符串
    """
    session_id = str(uuid.uuid4())[:8]
    print(f"[{session_id}] 开始处理音频: {audio_path}")

    ctx = new_context(method="asr.recognize")
    client = ASRClient(ctx=ctx)

    has_ffmpeg = check_ffmpeg()
    print(f"[{session_id}] ffmpeg可用: {has_ffmpeg}")

    local_file = None
    temp_files = []

    try:
        # 步骤1: 获取本地音频文件
        if audio_path.startswith("http://") or audio_path.startswith("https://"):
            local_file = f"/tmp/audio_download_{session_id}.raw"
            ok, msg = download_file(audio_path, local_file)
            if not ok:
                return json.dumps({"error": f"音频下载失败: {msg}"})
            temp_files.append(local_file)
            print(f"[{session_id}] {msg}")
        else:
            if not os.path.exists(audio_path):
                return json.dumps({"error": f"文件不存在: {audio_path}"})
            local_file = audio_path

        # 步骤2: 检查文件
        file_size_mb = get_file_size_mb(local_file)
        if file_size_mb > 100:
            return json.dumps({"error": f"文件过大: {file_size_mb:.1f}MB > 100MB限制"})
        print(f"[{session_id}] 文件大小: {file_size_mb:.1f}MB")

        # 步骤3: 用ffmpeg转码为标准WAV（关键步骤，解决ASR格式兼容问题）
        input_for_asr = local_file
        conversion_log = "未使用ffmpeg转码"

        if has_ffmpeg:
            converted_path = f"/tmp/audio_converted_{session_id}.wav"
            ok, detail = convert_audio_with_ffmpeg(local_file, converted_path)
            conversion_log = detail

            if ok:
                conv_size = get_file_size_mb(converted_path)
                print(f"[{session_id}] ✅ ffmpeg转码成功: {converted_path} ({conv_size:.1f}MB)")
                input_for_asr = converted_path
                temp_files.append(converted_path)
            else:
                print(f"[{session_id}] ❌ ffmpeg转码失败: {detail}，使用原始文件")
        else:
            print(f"[{session_id}] ffmpeg不可用，使用原始文件（ASR可能不支持某些格式）")

        # 步骤4: 读取音频并发送到ASR
        with open(input_for_asr, "rb") as f:
            audio_data = f.read()

        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        b64_size_mb = len(audio_base64) / (1024 * 1024)
        print(f"[{session_id}] 发送到ASR: {input_for_asr} (base64: {b64_size_mb:.1f}MB, 转码状态: {conversion_log})")

        text, data = client.recognize(uid=uid, base64_data=audio_base64)

        # 步骤5: 构建结果
        duration = data.get("result", {}).get("duration", 0)
        utterances = data.get("result", {}).get("utterances", [])

        result = {
            "full_text": text,
            "duration": duration,
            "duration_seconds": duration / 1000 if duration else 0,
            "segments": [],
            "conversion_log": conversion_log
        }

        for utterance in utterances:
            result["segments"].append({
                "text": utterance.get("text", ""),
                "start_time": utterance.get("start_time", 0),
                "end_time": utterance.get("end_time", 0)
            })

        print(f"[{session_id}] ASR完成: {len(utterances)}段, {duration/1000:.1f}秒")
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        error_msg = str(e)
        print(f"[{session_id}] ASR异常: {error_msg}")

        # 根据异常信息提供精准提示
        hint = "未知错误"
        e_lower = error_msg.lower()

        if "21109" in error_msg:
            hint = ("ASR服务端处理音频时出现内部错误（错误码21109）。"
                    "工具已自动用ffmpeg转码为16kHz/16bit/mono WAV标准格式，"
                    "但仍处理失败。请检查音频文件是否有实际人声内容，"
                    "或尝试截取较短片段重新处理。")
        elif "11103" in error_msg or "convert" in e_lower:
            hint = "音频格式转换失败，请尝试用ffmpeg提前转码为16kHz/16bit/mono WAV"
        elif "timeout" in e_lower:
            hint = "音频处理超时，请缩小音频文件或分段处理"
        elif "size" in e_lower or "limit" in e_lower:
            hint = "文件超过大小限制（100MB）"

        return json.dumps({
            "error": f"ASR识别失败: {error_msg}",
            "hint": hint,
            "conversion_log": conversion_log if 'conversion_log' in dir() else "N/A"
        })

    finally:
        # 清理临时文件（保留原始用户文件）
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"[{session_id}] 已清理临时文件: {f}")
            except:
                pass


def format_timestamp(ms: int) -> str:
    """将毫秒转换为 HH:MM:SS 格式"""
    seconds = ms // 1000
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"