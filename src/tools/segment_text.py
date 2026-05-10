"""
文本分段工具 - 将长文本按语义或标点切分成短句/短段
"""
import re
import json
from typing import List, Dict, Any
from langchain.tools import tool


@tool
def segment_text(text: str, min_length: int = 5, max_length: int = 200) -> str:
    """
    将长文本按标点符号切分成短句/短段
    
    Args:
        text: 需要分段的长文本
        min_length: 每个片段的最短字符数（过滤掉太短的片段）
        max_length: 每个片段的最大字符数（超过则继续拆分）
    
    Returns:
        JSON格式的分段列表，每个片段包含文本和位置信息
    """
    # 按句子结束标点分割
    # 句子结束符：。！？!?；;
    sentences = re.split(r'[。！？!?；;]', text)
    
    segments = []
    current_segment = ""
    segment_start = 0
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # 如果当前句子加上已有的文本超过max_length
        if len(current_segment) + len(sentence) > max_length and current_segment:
            # 保存当前段落
            if len(current_segment) >= min_length:
                segments.append({
                    "text": current_segment,
                    "start_char": segment_start,
                    "end_char": segment_start + len(current_segment)
                })
            # 开始新段落
            current_segment = sentence
            segment_start = text.find(sentence, segment_start)
        else:
            # 添加到当前段落
            if current_segment:
                current_segment += "。" + sentence
            else:
                current_segment = sentence
    
    # 保存最后一个段落
    if current_segment and len(current_segment) >= min_length:
        segments.append({
            "text": current_segment,
            "start_char": segment_start,
            "end_char": segment_start + len(current_segment)
        })
    
    return json.dumps(segments, ensure_ascii=False, indent=2)


@tool
def segment_with_timestamps(full_text: str, timestamps: List[Dict[str, Any]]) -> str:
    """
    根据ASR返回的时间戳信息进行分段
    
    Args:
        full_text: 完整的转写文本
        timestamps: ASR返回的utterances时间戳列表，JSON格式
    
    Returns:
        JSON格式的分段列表，每个片段包含文本、开始时间、结束时间
    """
    import json
    
    ts_list = json.loads(timestamps) if isinstance(timestamps, str) else timestamps
    
    segments = []
    for ts in ts_list:
        text = ts.get("text", "").strip()
        if text and len(text) >= 3:  # 过滤掉太短的片段
            segments.append({
                "text": text,
                "start_time": ts.get("start_time", 0),
                "end_time": ts.get("end_time", 0),
                "start_formatted": ms_to_time(ts.get("start_time", 0)),
                "end_formatted": ms_to_time(ts.get("end_time", 0))
            })
    
    return json.dumps(segments, ensure_ascii=False, indent=2)


def ms_to_time(ms: int) -> str:
    """毫秒转时间格式 HH:MM:SS"""
    seconds = ms // 1000
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def smart_segment(text: str) -> List[str]:
    """
    智能分段 - 考虑语义连贯性
    简单版本：按标点+换行分段
    """
    result = segment_text.invoke({"text": text, "min_length": 5, "max_length": 200})
    segments = json.loads(result)
    return [s["text"] for s in segments]
