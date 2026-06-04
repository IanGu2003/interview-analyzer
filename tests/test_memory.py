"""Tests for InterviewMemory module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))

from utils.memory import InterviewMemory, get_memory, reset_memory


class TestInterviewMemory:
    """Test InterviewMemory CRUD operations"""

    def test_store_and_retrieve_response(self):
        mem = InterviewMemory()
        mem.store_response("Q1", "我喜欢玩角色扮演游戏", confidence=0.95)
        responses = mem.get_responses_by_question("Q1")
        assert len(responses) == 1
        assert responses[0]["text"] == "我喜欢玩角色扮演游戏"
        assert responses[0]["confidence"] == 0.95

    def test_multiple_responses_same_question(self):
        mem = InterviewMemory()
        mem.store_response("Q1", "第一个回答", confidence=0.8)
        mem.store_response("Q1", "第二个回答", confidence=0.6)
        responses = mem.get_responses_by_question("Q1")
        assert len(responses) == 2

    def test_get_nonexistent_question(self):
        mem = InterviewMemory()
        responses = mem.get_responses_by_question("Q999")
        assert responses == []

    def test_get_all_responses(self):
        mem = InterviewMemory()
        mem.store_response("Q1", "回答1", confidence=0.9)
        mem.store_response("Q2", "回答2", confidence=0.8)
        all_resp = mem.get_all_responses()
        assert "Q1" in all_resp
        assert "Q2" in all_resp
        assert len(all_resp) == 2

    def test_unmatched_entries(self):
        mem = InterviewMemory()
        mem.add_unmatched("一些闲聊内容", "与问题无关")
        mem.add_unmatched("技术讨论", "不属于访谈范围")
        assert len(mem.unmatched) == 2
        assert mem.unmatched[0]["text"] == "一些闲聊内容"

    def test_fuzzy_match(self):
        mem = InterviewMemory()
        mem.add_fuzzy_match("关于游戏的讨论", "Q1", 0.75)
        assert len(mem.fuzzy_match) == 1
        assert mem.fuzzy_match[0]["matched_question_id"] == "Q1"

    def test_reset(self):
        mem = InterviewMemory()
        mem.store_response("Q1", "回答", confidence=0.9)
        mem.add_unmatched("无关内容")
        mem.reset()
        assert mem.get_all_responses() == {}
        assert mem.unmatched == []
        assert mem.fuzzy_match == []

    def test_full_memory_stats(self):
        mem = InterviewMemory()
        mem.store_response("Q1", "回答1", confidence=0.9)
        mem.store_response("Q1", "回答2", confidence=0.7)
        mem.store_response("Q2", "回答3", confidence=0.8)
        mem.add_unmatched("无关")
        full = mem.get_full_memory()
        assert full["stats"]["total_responses"] == 3
        assert full["stats"]["answered_questions"] == 2
        assert full["stats"]["unmatched_count"] == 1

    def test_global_memory_singleton(self):
        reset_memory()
        m1 = get_memory()
        m2 = get_memory()
        assert m1 is m2  # same instance
        m1.store_response("Q1", "测试", confidence=0.5)
        assert "Q1" in m2.get_all_responses()

    def test_confidence_rounding(self):
        mem = InterviewMemory()
        mem.store_response("Q1", "测试", confidence=0.876543)
        resp = mem.get_responses_by_question("Q1")
        assert resp[0]["confidence"] == 0.88

    def test_stored_at_timestamp(self):
        mem = InterviewMemory()
        mem.store_response("Q1", "测试", confidence=0.5)
        resp = mem.get_responses_by_question("Q1")
        assert "stored_at" in resp[0]
        assert "T" in resp[0]["stored_at"]  # ISO format

    def test_original_text_preserved(self):
        mem = InterviewMemory()
        mem.store_response("Q1", "清理后的文本", confidence=0.9,
                           original_text="原始带语气词的文本嗯...")
        resp = mem.get_responses_by_question("Q1")
        assert resp[0]["original_text"] == "原始带语气词的文本嗯..."
        assert resp[0]["text"] == "清理后的文本"