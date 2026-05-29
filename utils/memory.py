"""Interview memory management - stores question-answer mappings"""
from datetime import datetime


class InterviewMemory:
    """Manages interview question-answer responses"""

    def __init__(self):
        self.responses: dict[str, list[dict]] = {}
        self.unmatched: list[dict] = []
        self.fuzzy_match: list[dict] = []

    def store_response(self, question_id: str, text: str, confidence: float = 0.8,
                       original_text: str = ""):
        if question_id not in self.responses:
            self.responses[question_id] = []
        self.responses[question_id].append({
            "text": text,
            "confidence": round(confidence, 2),
            "original_text": original_text or text,
            "stored_at": datetime.now().isoformat(),
        })

    def get_responses_by_question(self, question_id: str) -> list[dict]:
        return self.responses.get(question_id, [])

    def get_all_responses(self) -> dict:
        return self.responses

    def add_unmatched(self, text: str, reason: str = ""):
        self.unmatched.append({"text": text, "reason": reason})

    def add_fuzzy_match(self, text: str, matched_question_id: str, score: float):
        self.fuzzy_match.append({
            "text": text,
            "matched_question_id": matched_question_id,
            "score": score,
        })

    def reset(self):
        self.responses = {}
        self.unmatched = []
        self.fuzzy_match = []

    def get_full_memory(self) -> dict:
        return {
            "responses": self.responses,
            "unmatched": self.unmatched,
            "fuzzy_match": self.fuzzy_match,
            "stats": {
                "total_responses": sum(len(v) for v in self.responses.values()),
                "answered_questions": len(self.responses),
                "unmatched_count": len(self.unmatched),
            }
        }


# Global instance
_global_memory: InterviewMemory | None = None


def get_memory() -> InterviewMemory:
    global _global_memory
    if _global_memory is None:
        _global_memory = InterviewMemory()
    return _global_memory


def reset_memory() -> InterviewMemory:
    global _global_memory
    _global_memory = InterviewMemory()
    return _global_memory