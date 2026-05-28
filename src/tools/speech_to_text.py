"""
音频转写工具 - 将音频文件转换为文字
- 支持本地音频文件和URL
- 自动使用ffmpeg转码为标准格式（16kHz, 16bit, mono WAV/MP3）
- 解决服务端audio convert failed问题
"""
import base64
import os
import json
import subprocess
import tempfile
from typing import List, Dict, Any, Optional
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
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def get_file_size_mb(path: str) -> float:
    """获取文件大小（MB）"""
    return os.path.getsize(path) / (1024 * 1024)


def convert_audio_to_standard(input_path: str, output_path: str) -> bool:
    """
    使用ffmpeg将音频转换为标准格式
    
    标准参数：
    - 采样率: 16kHz
    - 声道: mono
    - 编码: pcm_s16le (WAV)
    - 输出格式: WAV
    
    Returns:
        True表示转换成功
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", TARGET_SAMPLE_RATE,
            "-ac", TARGET_CHANNELS,
            "-sample_fmt", "s16",
            "-f", "wav",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"[WARN] ffmpeg convert failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("[WARN] ffmpeg convert timeout")
        return False
    except Exception as e:
        print(f"[WARN] ffmpeg convert exception: {e}")
        return False


def probe_audio_info(path: str) -> Dict[str, Any]:
    """使用ffprobe获取音频信息"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except:
        pass
    return {}


def download_file(url: str, local_path: str, max_size_mb: int = 100) -> bool:
    """从URL下载文件到本地"""
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        
        # 流式写入并检查大小
        downloaded = 0
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded > max_size_mb * 1024 * 1024:
                    os.remove(local_path)
                    print(f"[ERROR] 文件超过{max_size_mb}MB限制")
                    return False
        return True
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        return False


@tool
def speech_to_text(audio_path: str, uid: str = "interview_user") -> str:
    """
    将音频文件转换为文字（支持 .mp3, .wav, .ogg, .m4a 等格式）
    支持本地路径和HTTP/HTTPS URL
    自动使用ffmpeg将音频转为标准格式再提交，解决转码失败问题
    
    Args:
        audio_path: 音频文件路径或URL
        uid: 用户唯一标识符
    
    Returns:
        包含转写文字和时间戳信息的JSON字符串
    """
    ctx = new_context(method="asr.recognize")
    client = ASRClient(ctx=ctx)
    
    has_ffmpeg = check_ffmpeg()
    local_file = None
    need_cleanup = False
    
    try:
        # 步骤1: 获取本地音频文件
        if audio_path.startswith("http://") or audio_path.startswith("https://"):
            # 从URL下载到临时文件
            local_file = tempfile.mktemp(suffix=".tmp_audio")
            success = download_file(audio_path, local_file)
            if not success:
                return json.dumps({
                    "error": "音频文件下载失败",
                    "hint": "请检查URL是否可访问，或文件是否超过100MB"
                })
            need_cleanup = True
            print(f"[INFO] 已从URL下载音频到: {local_file}")
        else:
            local_file = audio_path
        
        # 步骤2: 检查文件是否存在
        if not os.path.exists(local_file):
            return json.dumps({
                "error": f"文件不存在: {local_file}",
                "hint": "请检查音频文件路径是否正确"
            })
        
        # 步骤3: 检查文件大小（ASR限制100MB）
        file_size_mb = get_file_size_mb(local_file)
        if file_size_mb > 100:
            return json.dumps({
                "error": f"文件过大: {file_size_mb:.1f}MB，超过100MB限制",
                "hint": "请压缩或分段处理音频"
            })
        print(f"[INFO] 音频文件大小: {file_size_mb:.1f}MB")
        
        # 步骤4: 如果ffmpeg可用，先转码为标准格式
        input_for_asr = local_file
        
        if has_ffmpeg:
            # 探测原音频信息
            probe_info = probe_audio_info(local_file)
            if probe_info:
                streams = probe_info.get("streams", [])
                if streams:
                    s = streams[0]
                    print(f"[INFO] 原音频: codec={s.get('codec_name')}, "
                          f"sample_rate={s.get('sample_rate')}, "
                          f"channels={s.get('channels')}")
            
            # 转码为标准格式
            converted_file = tempfile.mktemp(suffix=".wav")
            convert_ok = convert_audio_to_standard(local_file, converted_file)
            
            if convert_ok:
                converted_size = get_file_size_mb(converted_file)
                print(f"[INFO] ffmpeg转码成功: {converted_size:.1f}MB -> 16kHz/mono/16bit WAV")
                
                if converted_size <= 100:
                    input_for_asr = converted_file
                    if need_cleanup:
                        # 如果是从URL下载的，删除原始文件
                        try: os.remove(local_file)
                        except: pass
                    local_file = converted_file
                    need_cleanup = True
                else:
                    print(f"[WARN] 转码后文件仍超过100MB({converted_size:.1f}MB)，使用原始文件")
                    try: os.remove(converted_file)
                    except: pass
            else:
                print("[WARN] ffmpeg转码失败，使用原始文件")
                try: os.remove(converted_file)
                except: pass
        else:
            print("[INFO] ffmpeg不可用，直接使用原始文件（ASR服务端可能不支持某些格式）")
        
        # 步骤5: 读取音频并发送到ASR
        with open(input_for_asr, "rb") as f:
            audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        
        print(f"[INFO] 正在发送音频到ASR服务 (base64大小: {len(audio_base64)//1024}KB)...")
        text, data = client.recognize(uid=uid, base64_data=audio_base64)
        
        # 步骤6: 构建结果
        duration = data.get("result", {}).get("duration", 0)
        
        result = {
            "full_text": text,
            "duration": duration,
            "duration_seconds": duration / 1000 if duration else 0,
            "segments": []
        }
        
        utterances = data.get("result", {}).get("utterances", [])
        for utterance in utterances:
            result["segments"].append({
                "text": utterance.get("text", ""),
                "start_time": utterance.get("start_time", 0),
                "end_time": utterance.get("end_time", 0)
            })
        
        print(f"[INFO] ASR识别完成: {len(utterances)}段, 时长{duration/1000:.1f}秒")
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] ASR识别异常: {error_msg}")
        
        # 提供更有帮助的错误提示
        hint = "未知错误"
        if "timeout" in error_msg.lower():
            hint = "音频处理超时，请缩小音频文件或分段处理"
        elif "convert" in error_msg.lower() or "11103" in error_msg:
            hint = "音频格式转换失败，请尝试用ffmpeg提前转码为16kHz/16bit/mono WAV"
        elif "size" in error_msg.lower() or "limit" in error_msg.lower():
            hint = "文件超过大小限制（100MB）"
        
        return json.dumps({
            "error": f"ASR识别失败: {error_msg}",
            "hint": hint
        })
        
    finally:
        # 清理临时文件
        if need_cleanup and local_file and os.path.exists(local_file):
            try:
                os.remove(local_file)
            except:
                pass


def format_timestamp(ms: int) -> str:
    """将毫秒转换为 HH:MM:SS 格式"""
    seconds = ms // 1000
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"