"""
音频转写工具 - 将音频文件转换为文字
"""
import base64
import os
from typing import List, Dict, Any
from langchain.tools import tool
from coze_coding_dev_sdk import ASRClient
from coze_coding_utils.runtime_ctx.context import new_context
import requests


@tool
def speech_to_text(audio_path: str, uid: str = "interview_user") -> str:
    """
    将音频文件转换为文字（支持 .mp3, .wav, .ogg, .m4a 格式）
    
    Args:
        audio_path: 音频文件路径，支持本地路径或URL
        uid: 用户唯一标识符
    
    Returns:
        包含转写文字和时间戳信息的JSON字符串
    """
    ctx = new_context(method="asr.recognize")
    client = ASRClient(ctx=ctx)
    
    # 判断是本地文件还是URL
    if audio_path.startswith("http://") or audio_path.startswith("https://"):
        # URL模式
        text, data = client.recognize(uid=uid, url=audio_path)
    else:
        # 本地文件模式，需要转换为base64
        with open(audio_path, "rb") as f:
            audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        text, data = client.recognize(uid=uid, base64_data=audio_base64)
    
    # 构建带时间戳的结果
    result = {
        "full_text": text,
        "duration": data.get("result", {}).get("duration", 0),
        "segments": []
    }
    
    # 提取每个句子的时间戳
    utterances = data.get("result", {}).get("utterances", [])
    for utterance in utterances:
        result["segments"].append({
            "text": utterance.get("text", ""),
            "start_time": utterance.get("start_time", 0),
            "end_time": utterance.get("end_time", 0)
        })
    
    import json
    return json.dumps(result, ensure_ascii=False, indent=2)


def format_timestamp(ms: int) -> str:
    """将毫秒转换为 HH:MM:SS 格式"""
    seconds = ms // 1000
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
