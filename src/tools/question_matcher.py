"""
问题匹配工具 - 将文本片段匹配到最接近的结构化问题
"""
import json
from typing import List, Dict, Any, Tuple
from langchain.tools import tool
from coze_coding_dev_sdk import LLMClient
from coze_coding_utils.runtime_ctx.context import new_context
from langchain_core.messages import SystemMessage, HumanMessage


# 问题匹配的系统提示词
MATCHER_SYSTEM_PROMPT = """你是一个专业的访谈内容分析助手。你的任务是将受访者的回答片段匹配到最合适的访谈问题。

注意事项：
1. 如果回答是对某个问题的直接回应，匹配该问题
2. 如果回答暗示或比喻性地回答了某个问题，也要匹配
3. 如果回答明显与所有问题都不相关，返回"unmatched"
4. 评分范围0.0-1.0，越高表示匹配度越高

只输出JSON格式：
{"question_id": "Q1"或"unmatched", "score": 0.0-1.0, "matched_aspect": "匹配的具体方面"}"""


@tool
def question_matcher(text: str, questions: str) -> str:
    """
    将文本片段匹配到最接近的访谈问题
    
    Args:
        text: 需要匹配的受访者回答片段
        questions: 访谈问题列表，JSON格式
            示例: [{"question_id": "Q1", "question_text": "...", "possible_probes": ["..."]}]
    
    Returns:
        JSON格式的匹配结果，包含最佳匹配的问题ID和评分
    """
    ctx = new_context(method="llm.invoke")
    client = LLMClient(ctx=ctx)
    
    # 解析问题列表
    try:
        q_list = json.loads(questions) if isinstance(questions, str) else questions
        if not isinstance(q_list, list):
            return json.dumps({"question_id": "unmatched", "score": 0.0, "matched_aspect": "格式错误"})
    except Exception as e:
        return json.dumps({"question_id": "unmatched", "score": 0.0, "matched_aspect": f"解析错误: {str(e)}"})
    
    # 构建问题列表文本
    q_text = "\n".join([
        f"- {q.get('question_id', f'Q{i+1}')}: {q.get('question_text', q)}"
        f"{' (追问: ' + ', '.join(q.get('possible_probes', [])) + ')' if q.get('possible_probes') else ''}"
        for i, q in enumerate(q_list)
    ])
    
    user_prompt = f"""访谈问题列表：
{q_text}

受访者回答片段：
{text}

请判断这个回答最匹配哪个问题？"""
    
    messages = [
        SystemMessage(content=MATCHER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]
    
    response = client.invoke(messages=messages, temperature=0.1)
    
    # 解析LLM返回的JSON
    try:
        result = json.loads(response.content)
    except:
        content = response.content if isinstance(response.content, str) else str(response.content)
        import re
        match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = {"question_id": "unmatched", "score": 0.0, "matched_aspect": "解析失败"}
    
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def batch_question_matcher(segments: str, questions: str) -> str:
    """
    批量匹配多个文本片段到问题
    
    Args:
        segments: 文本片段列表，JSON格式 [{"text": "...", "start_time": 1234}]
        questions: 访谈问题列表，JSON格式
    
    Returns:
        每个片段的匹配结果列表
    """
    import json
    
    seg_list = json.loads(segments) if isinstance(segments, str) else segments
    results = []
    
    for seg in seg_list:
        seg_text = seg if isinstance(seg, str) else seg.get("text", "")
        start_time = seg.get("start_time", 0) if isinstance(seg, dict) else 0
        
        match_result = question_matcher.invoke({"text": seg_text, "questions": questions})
        match_data = json.loads(match_result)
        
        results.append({
            "text": seg_text,
            "start_time": start_time,
            "matched_question_id": match_data.get("question_id", "unmatched"),
            "match_score": match_data.get("score", 0.0),
            "matched_aspect": match_data.get("matched_aspect", "")
        })
    
    return json.dumps(results, ensure_ascii=False, indent=2)


def get_best_match(text: str, questions: List[Dict]) -> Tuple[str, float]:
    """
    快速获取最佳匹配（返回元组）
    
    Args:
        text: 文本片段
        questions: 问题列表
    
    Returns:
        (question_id, score) 元组
    """
    result = question_matcher.invoke({"text": text, "questions": json.dumps(questions)})
    data = json.loads(result)
    return data.get("question_id", "unmatched"), data.get("score", 0.0)
