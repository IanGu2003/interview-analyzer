"""Telemetry & Observability for Interview Analyzer

Records every LLM call as JSON Lines to logs/events.jsonl.
Supports contract verification (quote_verified) to detect LLM hallucinations.
"""
import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── Log Path ────────────────────────────────────────────────────────

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_EVENTS_PATH = os.path.join(_LOG_DIR, "events.jsonl")


def _ensure_log_dir():
    os.makedirs(_LOG_DIR, exist_ok=True)


def log_event(event: dict):
    """Write a single event as JSON Line to logs/events.jsonl"""
    _ensure_log_dir()
    event["timestamp"] = event.get(
        "timestamp", datetime.now(timezone.utc).isoformat()
    )
    try:
        with open(_EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write telemetry event: {e}")


# ── Telemetry Wrapper ────────────────────────────────────────────────


def try_parse_json(text: str) -> bool:
    """Check if text is valid JSON (array or object)"""
    stripped = text.strip()
    # Strip markdown code blocks
    if "```json" in stripped:
        stripped = stripped.split("```json")[1]
        if "```" in stripped:
            stripped = stripped.split("```")[0]
    elif "```" in stripped:
        stripped = stripped.split("```")[1]
        if "```" in stripped:
            stripped = stripped.split("```")[0]
    stripped = stripped.strip()
    try:
        if stripped.startswith("[") or stripped.startswith("{"):
            json.loads(stripped)
            return True
        return False
    except json.JSONDecodeError:
        return False


def llm_chat_with_telemetry(
    client: OpenAI,
    messages: list,
    model: str = "gpt-4o",
    call_type: str = "unknown",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    metadata: Optional[dict] = None,
) -> str:
    """LLM call wrapped with telemetry: logs latency, tokens, JSON parse status.

    Args:
        client: OpenAI-compatible client
        messages: Chat messages list
        model: Model name
        call_type: Identifier for the call stage
            (e.g. 'extract_answers', 'code_response', 'clean_text')
        temperature: LLM temperature
        max_tokens: Max output tokens
        metadata: Extra context to log (e.g. question_id, input_length)

    Returns:
        The LLM response text (same as direct call)
    """
    start = time.time()
    input_length = len(messages[-1]["content"]) if messages else 0
    error = None
    result = ""
    total_tokens = 0

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = resp.choices[0].message.content or ""
        total_tokens = resp.usage.total_tokens if resp.usage else 0
    except Exception as e:
        error = str(e)
        raise
    finally:
        latency_ms = (time.time() - start) * 1000
        event = {
            "call_type": call_type,
            "model": model,
            "input_length": input_length,
            "output_length": len(result),
            "latency_ms": round(latency_ms, 1),
            "tokens": total_tokens,
            "json_parse_ok": try_parse_json(result) if result else False,
            "has_error": error is not None,
        }
        if error:
            event["error"] = error[:500]
        if metadata:
            event.update(metadata)
        log_event(event)

    return result


# ── Contract Verification ────────────────────────────────────────────


def verify_quote_in_transcript(
    extracted_items: list[dict], transcript: str
) -> list[dict]:
    """Verify each extracted answer's original_text appears in transcript.

    Adds 'quote_verified' field to each item.
    quote_verified=False indicates potential LLM hallucination.

    Returns the same list with added field.
    """
    for item in extracted_items:
        original = item.get("original_text", "")
        if original:
            item["quote_verified"] = original in transcript
        else:
            item["quote_verified"] = False

    # Log aggregate hallucination stats
    total = len(extracted_items)
    verified = sum(1 for i in extracted_items if i.get("quote_verified"))
    log_event({
        "call_type": "contract_check.extract_answers",
        "extracted_count": total,
        "verified_count": verified,
        "hallucinated_count": total - verified,
        "hallucination_rate": round((total - verified) / total, 3) if total > 0 else 0,
    })

    return extracted_items


def verify_quote_in_answer(
    coded_result: dict, answer_text: str
) -> dict:
    """Verify the coded result's source_quote appears in the answer text.

    Adds 'quote_verified' field to the result.
    """
    source_quote = coded_result.get("source_quote", "")
    if source_quote:
        coded_result["quote_verified"] = source_quote in answer_text
    else:
        coded_result["quote_verified"] = True  # No quote to verify

    return coded_result