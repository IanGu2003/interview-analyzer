"""
相关性判断工具 - 判断文本片段是否与访谈主题相关
"""
import json
from typing import List, Dict, Any
from langchain.tools import tool
from coze_coding_dev_sdk import LLMClient
from coze_coding_utils.runtime_ctx.context import new_context
from langchain_core.messages import SystemMessage, HumanMessage


# 相关性判断的系统提示词
RELEVANCE_SYSTEM_PROMPT = """你是一个专业的访谈内容分析助手。你的任务是判断给定的文本片段是否与访谈主题相关。

评分标准：
- 1.0: 完全相关，是受访者对访谈问题的直接回答或补充说明
- 0.7-0.9: 比较相关，与访谈主题有较大关联
- 0.4-0.6: 部分相关，可能涉及但不是核心内容
- 0.1-0.3: 基本不相关，可能是跑题或闲聊
- 0.0: 完全不相关，与访谈毫无关系

只输出JSON格式：
{"score": 0.0-1.0, "reason": "简短理由"}"""


@tool
def relevance_classifier(text: str, questions: str = None) -> str:
    """
    判断文本片段是否与访谈主题相关
    
    Args:
        text: 需要判断的文本片段
        questions: 访谈问题列表，JSON格式。如果为None，则只做通用相关性判断
    
    Returns:
        JSON格式的相关性评分和理由
    """
    ctx = new_context(method="llm.invoke")
    client = LLMClient(ctx=ctx)
    
    # 构建提示词
    if questions:
        try:
            q_list = json.loads(questions) if isinstance(questions, str) else questions
            if isinstance(q_list, list) and len(q_list) > 0:
                q_text = "\n".join([f"- {q.get('question_text', q) if isinstance(q, dict) else q}" 
                                   for q in q_list[:5]])  # 只显示前5个问题
                user_prompt = f"""访谈主题相关问题（部分）：
{q_text}

待判断文本：
{text}

判断这段话是否与访谈主题相关？"""
            else:
                user_prompt = f"""待判断文本：
{text}

判断这段话是否与访谈主题相关？"""
        except:
            user_prompt = f"""待判断文本：
{text}

判断这段话是否与访谈主题相关？"""
    else:
        user_prompt = f"""待判断文本：
{text}

判断这段话是否与访谈主题相关？"""
    
    messages = [
        SystemMessage(content=RELEVANCE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]
    
    response = client.invoke(messages=messages, temperature=0.1)
    
    # 解析LLM返回的JSON
    try:
        result = json.loads(response.content)
    except:
        # 如果解析失败，尝试提取JSON部分
        content = response.content if isinstance(response.content, str) else str(response.content)
        import re
        match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = {"score": 0.5, "reason": "评估失败"}
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def batch_relevance_classifier(segments: str, questions: str = None) -> str:
    """
    批量判断多个文本片段的相关性
    
    Args:
        segments: 文本片段列表，JSON格式
        questions: 访谈问题列表，JSON格式
    
    Returns:
        每个片段的相关性评分列表，JSON格式
    """
    import json
    
    seg_list = json.loads(segments) if isinstance(segments, str) else segments
    
    results = []
    for seg in seg_list:
        seg_text = seg if isinstance(seg, str) else seg.get("text", "")
        result = relevance_classifier.invoke({"text": seg_text, "questions": questions})
        results.append({
            "original": seg_text,
            "relevance": json.loads(result)
        })
    
    return json.dumps(results, ensure_ascii=False, indent=2)


def is_relevant(text: str, questions: List[str], threshold: float = 0.4) -> bool:
    """
    快速判断文本是否相关（返回布尔值）
    
    Args:
        text: 文本片段
        questions: 问题列表
        threshold: 阈值，默认0.4
    
    Returns:
        True表示相关，False表示不相关
    """
    result = relevance_classifier.invoke({"text": text, "questions": json.dumps(questions)})
    data = json.loads(result)
    return data.get("score", 0) >= threshold
