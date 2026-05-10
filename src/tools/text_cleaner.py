"""
文本清理工具 - 剔除语气词，保留原话主干
"""
import re
from typing import List
from langchain.tools import tool


# 默认语气词列表（中文口语中常见的填充词）
DEFAULT_FILLER_WORDS = [
    "嗯", "啊", "那个", "就是说", "然后", "这个", "那个",
    "呃", "哦", "哈", "哎", "诶", "唉", "哎呀", "呃呃",
    "是吧", "对吧", "我觉得", "就是", "基本上", "大概",
    "可能", "应该", "好像", "其实", "说实话", "老实说",
    "你知道吗", "我说", "你看", "这样的话", "那么",
    "不过", "但是", "而且", "或者", "或者说"
]


@tool
def text_cleaner(text: str, filler_words: str = None) -> str:
    """
    清理文本中的语气词，保留原话主干
    
    Args:
        text: 需要清理的原始文本
        filler_words: 自定义语气词列表，逗号分隔。如果为None则使用默认列表
    
    Returns:
        清理后的文本
    """
    # 解析自定义语气词列表
    if filler_words:
        word_list = [w.strip() for w in filler_words.split(",") if w.strip()]
    else:
        word_list = DEFAULT_FILLER_WORDS
    
    cleaned_text = text
    
    # 移除语气词（使用多种模式）
    for word in word_list:
        # 移除独立出现的语气词（前后有空格或标点）
        pattern = rf'\s*{re.escape(word)}\s*'
        cleaned_text = re.sub(pattern, ' ', cleaned_text)
        
        # 移除句首的语气词
        pattern = rf'^{re.escape(word)}\s*'
        cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # 移除句末的语气词
        pattern = rf'\s*{re.escape(word)}$'
        cleaned_text = re.sub(pattern, '', cleaned_text)
    
    # 清理多余的空格
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


@tool
def batch_text_cleaner(texts: List[str], filler_words: str = None) -> str:
    """
    批量清理多个文本片段
    
    Args:
        texts: 文本列表（JSON格式的字符串）
        filler_words: 自定义语气词列表，逗号分隔
    
    Returns:
        清理后的文本列表（JSON格式）
    """
    import json
    
    text_list = json.loads(texts) if isinstance(texts, str) else texts
    cleaned_list = []
    
    for text in text_list:
        cleaned = text_cleaner.invoke({"text": text, "filler_words": filler_words})
        cleaned_list.append(cleaned)
    
    return json.dumps(cleaned_list, ensure_ascii=False, indent=2)


def extract_core_meaning(text: str) -> str:
    """
    使用规则方式提取核心语义（简单的语气词移除）
    这是text_cleaner的简化版本
    """
    return text_cleaner.invoke({"text": text})
