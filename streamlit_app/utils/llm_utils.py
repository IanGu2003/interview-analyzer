"""LLM utilities for interview analysis - using OpenAI-compatible API"""
import json
import re
from openai import OpenAI


def get_client(api_key: str, base_url: str = "https://api.openai.com/v1"):
    return OpenAI(api_key=api_key, base_url=base_url)


def llm_chat(
    client: OpenAI,
    messages: list,
    model: str = "gpt-4o",
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> str:
    """Simple LLM chat completion"""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def extract_answers_from_transcript(
    client: OpenAI,
    transcript: str,
    questions: list[dict],
    model: str = "gpt-4o",
) -> list[dict]:
    """Extract respondent answers from transcript and match to questions

    Returns list of dicts: {question_id, cleaned_text, score, original_text}
    """
    q_text = "\n".join(
        [f"{q['question_id']}: {q['question_text']}" for q in questions]
    )

    system_prompt = """你是一个专业的访谈分析助手。以下是游戏用户研究访谈的转写文本（单声道录音，主持人与受访者对话混合）。

任务：找出所有**受访者回答**的内容，匹配到对应问题，清理语气词。

规则：
1. 只提取受访者说的话，跳过主持人的提问、引导语、简短回应
2. 跳过"好的"、"哦"、"嗯"、"这样"等纯语调词
3. 受访者回答通常比较简短（1-20个字），不要因为简短就忽略
4. cleaned_text要清理语气词（嗯、啊、哦、呃、那个、就是说等）
5. score表示匹配置信度（0.0-1.0）

输出格式（纯JSON数组，不要其他文字）：
[
  {"question_id": "Q1", "cleaned_text": "清理后的回答", "score": 0.8, "original_text": "原文片段"},
  {"question_id": "Q2", ...}
]"""

    user_prompt = f"""访谈问题：
{q_text}

访谈转写全文：
{transcript}

请提取所有受访者回答并匹配到对应问题。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = llm_chat(client, messages, model=model, temperature=0.1, max_tokens=4000)

    # Parse JSON from response
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON array from text
        match = re.search(r'\[[\s\S]*\]', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return []


def code_response(
    client: OpenAI,
    question_text: str,
    answer_text: str,
    model: str = "gpt-4o",
) -> dict:
    """Perform preliminary coding on a single answer

    Returns dict with theme_code, sub_code, keywords, sentiment, memo
    """
    system_prompt = """你是一个质性研究编码助手。请对访谈回答进行初步编码。

编码维度：
1. 主题编码（一级编码）：概括回答所属的核心主题类别
   - 如：产品功能评价、情感态度、使用行为、竞品对比、推荐意愿等
2. 子编码（二级编码）：更具体的子类别
   - 如：功能缺陷、满意度、情感连接、使用频率、推荐理由等
3. 关键词：提取2-5个核心关键词
4. 情感倾向：正面/负面/中性/混合
5. 编码备注：简要说明编码依据

输出格式（纯JSON）：
{"theme_code": "一级编码", "sub_code": "二级编码", "keywords": ["关键词1", "关键词2"], "sentiment": "正面/负面/中性/混合", "memo": "编码依据说明"}"""

    user_prompt = f"""问题：{question_text}
回答：{answer_text}

请对这个回答进行初步编码。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = llm_chat(client, messages, model=model, temperature=0.1, max_tokens=1000)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {
        "theme_code": "未分类",
        "sub_code": "待确认",
        "keywords": [],
        "sentiment": "中性",
        "memo": "编码失败，请人工核对",
    }


def clean_text(client: OpenAI, text: str, model: str = "gpt-4o") -> str:
    """Clean filler words from text"""
    system_prompt = "你是一个文本清理助手。清理文本中的语气词填充词（嗯、啊、哦、呃、那个、就是说等），保留核心语义。仅返回清理后的文本。"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"清理语气词：{text}"},
    ]
    return llm_chat(client, messages, model=model, temperature=0.1, max_tokens=500)