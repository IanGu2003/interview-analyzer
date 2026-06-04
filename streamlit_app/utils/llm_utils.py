"""LLM utilities for interview analysis - using OpenAI-compatible API

Prompts are versioned separately in prompts/ directory.
See prompts/CHANGELOG.md for history of prompt changes.
"""
import os
import json
import re
from openai import OpenAI
from .knowledge_base import get_kb


# ── Prompt Versioning ──────────────────────────────────────────────

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_FALLBACK_CACHE: dict[str, str] = {}


def _prompt_fallback(name: str) -> str:
    """Hardcoded fallback prompts when file not found (for backward compat)"""
    fallbacks = {
        "v1_extract_answers": (
            "你是一个专业的访谈分析助手。以下是游戏用户研究访谈的转写文本（单声道录音，主持人与受访者对话混合）。\n\n"
            "任务：找出所有**受访者回答**的内容，匹配到对应问题，清理语气词。\n\n"
            "规则：\n"
            "1. 只提取受访者说的话，跳过主持人的提问、引导语、简短回应\n"
            "2. 跳过\"好的\"、\"哦\"、\"嗯\"、\"这样\"等纯语调词\n"
            "3. 受访者回答通常比较简短（1-20个字），不要因为简短就忽略\n"
            "4. cleaned_text要清理语气词（嗯、啊、哦、呃、那个、就是说等）\n"
            "5. score表示匹配置信度（0.0-1.0）\n\n"
            "输出格式（纯JSON数组，不要其他文字）：\n"
            '[\n'
            '  {"question_id": "Q1", "cleaned_text": "清理后的回答", "score": 0.8, "original_text": "原文片段"},\n'
            '  {"question_id": "Q2", ...}\n'
            "]"
        ),
        "v1_code_response": (
            "你是一个质性研究编码助手。请对访谈回答进行初步编码。\n\n"
            "编码维度：\n"
            "1. 主题编码（一级编码）：概括回答所属的核心主题类别\n"
            "   - 如：产品功能评价、情感态度、使用行为、竞品对比、推荐意愿等\n"
            "2. 子编码（二级编码）：更具体的子类别\n"
            "   - 如：功能缺陷、满意度、情感连接、使用频率、推荐理由等\n"
            "3. 关键词：提取2-5个核心关键词\n"
            "4. 情感倾向：正面/负面/中性/混合\n"
            "5. 编码备注：简要说明编码依据\n\n"
            "输出格式（纯JSON）：\n"
            '{"theme_code": "一级编码", "sub_code": "二级编码", "keywords": ["关键词1", "关键词2"], "sentiment": "正面/负面/中性/混合", "memo": "编码依据说明"}'
        ),
        "v1_clean_text": (
            "你是一个文本清理助手。清理文本中的语气词填充词（嗯、啊、哦、呃、那个、就是说等），"
            "保留核心语义。仅返回清理后的文本。"
        ),
    }
    return fallbacks.get(name, "")


def load_prompt(name: str, version: str = "v2") -> str:
    """Load a prompt from file, with fallback to hardcoded string
    
    Args:
        name: Prompt name (e.g. 'extract_answers', 'code_response', 'clean_text')
        version: Version tag (e.g. 'v1', 'v2')
    
    Returns:
        Prompt text as string
    """
    filepath = os.path.join(_PROMPT_DIR, version, f"{name}.txt")
    
    # Try file first
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    
    # Fallback to hardcoded
    cache_key = f"{version}_{name}"
    if cache_key not in _FALLBACK_CACHE:
        _FALLBACK_CACHE[cache_key] = _prompt_fallback(cache_key)
    return _FALLBACK_CACHE[cache_key]


def render_prompt(template: str, kb_context: str = "") -> str:
    """Simple template renderer for prompt with optional KB context
    
    Supports:
    - {% if kb_context %}...{{ kb_context }}...{% endif %}
    """
    if "{% if kb_context %}" in template:
        parts = template.split("{% if kb_context %}")
        before = parts[0]
        rest = parts[1]
        inner_and_after = rest.split("{% endif %}", 1)
        inner = inner_and_after[0]
        after = inner_and_after[1] if len(inner_and_after) > 1 else ""
        
        if kb_context:
            rendered_inner = inner.replace("{{ kb_context }}", kb_context)
            return before + rendered_inner + after
        else:
            return before + after
    return template


# ── Core LLM Functions ────────────────────────────────────────────


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


# ── Prompt Version: Semantic Matching ──────────────────────────────

_EXTRACT_PROMPT_VERSION = "v2"  # Change here to switch prompt versions


def extract_answers_from_transcript(
    client: OpenAI,
    transcript: str,
    questions: list[dict],
    model: str = "gpt-4o",
) -> list[dict]:
    """Extract respondent answers from transcript and match to questions (v1)"""
    q_text = "\n".join(
        [f"{q['question_id']}: {q['question_text']}" for q in questions]
    )

    system_prompt = load_prompt("extract_answers", "v1")

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

    return _parse_json_array(content)


def code_response(
    client: OpenAI,
    question_text: str,
    answer_text: str,
    model: str = "gpt-4o",
) -> dict:
    """Perform preliminary coding on a single answer (v1)"""
    system_prompt = load_prompt("code_response", "v1")

    user_prompt = f"""问题：{question_text}
回答：{answer_text}

请对这个回答进行初步编码。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = llm_chat(client, messages, model=model, temperature=0.1, max_tokens=1000)

    result = _parse_json_object(content)
    if result:
        return result
    return {
        "theme_code": "未分类",
        "sub_code": "待确认",
        "keywords": [],
        "sentiment": "中性",
        "memo": "编码失败，请人工核对",
    }


def clean_text(client: OpenAI, text: str, model: str = "gpt-4o") -> str:
    """Clean filler words from text"""
    system_prompt = load_prompt("clean_text", "v1")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"清理语气词：{text}"},
    ]
    return llm_chat(client, messages, model=model, temperature=0.1, max_tokens=500)


# ═══════════════════════════════════════════════════════════════
#  Knowledge-Base Enhanced Functions (v2 prompts)
# ═══════════════════════════════════════════════════════════════


def _build_kb_context(query_text: str, top_k: int = 5) -> str:
    """Build knowledge base context string for prompt injection"""
    kb = get_kb()
    if kb.count()["total"] == 0:
        return ""
    return kb.get_context(query_text, top_k=top_k, max_chars=3000)


def extract_answers_with_kb(
    client: OpenAI,
    transcript: str,
    questions: list[dict],
    model: str = "gpt-4o",
) -> list[dict]:
    """Extract answers from transcript with Knowledge Base enhancement (v2)"""
    q_text = "\n".join(
        [f"{q['question_id']}: {q['question_text']}" for q in questions]
    )

    # Build KB context
    kb_context = _build_kb_context(transcript + "\n" + q_text, top_k=6)

    # Load v2 template and render with KB context
    template = load_prompt("extract_answers", "v2")
    system_prompt = render_prompt(template, kb_context=kb_context)

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

    return _parse_json_array(content)


def code_response_with_kb(
    client: OpenAI,
    question_text: str,
    answer_text: str,
    model: str = "gpt-4o",
) -> dict:
    """Perform preliminary coding with Knowledge Base enhancement (v2)"""
    kb_context = _build_kb_context(
        f"{question_text} {answer_text}",
        top_k=4,
    )

    template = load_prompt("code_response", "v2")
    system_prompt = render_prompt(template, kb_context=kb_context)

    user_prompt = f"""问题：{question_text}
回答：{answer_text}

请对这个回答进行初步编码。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = llm_chat(client, messages, model=model, temperature=0.1, max_tokens=1000)

    result = _parse_json_object(content)
    if result:
        return result
    return {
        "theme_code": "未分类",
        "sub_code": "待确认",
        "keywords": [],
        "sentiment": "中性",
        "memo": "编码失败，请人工核对",
    }


# ═══════════════════════════════════════════════════════════════
#  JSON Parsing Utilities
# ═══════════════════════════════════════════════════════════════


def _parse_json_array(text: str) -> list:
    """Try to parse a JSON array from LLM response"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[[\s\S]*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return []


def _parse_json_object(text: str) -> dict | None:
    """Try to parse a JSON object from LLM response"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None