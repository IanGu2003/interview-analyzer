"""Tests for KnowledgeBase module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))

from utils.knowledge_base import KnowledgeBase


class TestKnowledgeBase:
    """Test KnowledgeBase CRUD and retrieval operations"""

    def test_add_text(self):
        kb = KnowledgeBase()
        count = kb.add_text("这是一段关于角色扮演游戏的用户访谈记录",
                            category="访谈记录", title="RPG用户访谈")
        assert count >= 1  # should create at least 1 chunk
        assert len(kb.documents) >= 1
        assert kb.documents[0]["category"] == "访谈记录"

    def test_add_glossary_entry(self):
        kb = KnowledgeBase()
        kb.add_glossary_entry("RPG", "角色扮演游戏，玩家扮演虚拟世界中的一个角色")
        assert len(kb.documents) == 1
        assert kb.documents[0]["category"] == "术语表"
        assert kb.documents[0]["source"] == "术语表"

    def test_empty_kb_search(self):
        kb = KnowledgeBase()
        results = kb.search("游戏", top_k=5)
        assert results == []

    def test_search_finds_relevant(self):
        kb = KnowledgeBase()
        kb.add_text("玩家在开放世界游戏中探索和完成任务", category="访谈记录")
        kb.add_text("竞技游戏的平衡性对玩家体验很重要", category="访谈记录")
        results = kb.search("开放世界", top_k=5)
        assert len(results) >= 1
        assert "开放世界" in results[0]["text"]

    def test_search_multilingual(self):
        kb = KnowledgeBase()
        kb.add_text("玩家喜欢在游戏中与朋友社交互动", category="访谈记录")
        kb.add_glossary_entry("MMO", "大型多人在线游戏")
        # Search in Chinese
        results = kb.search("社交", top_k=5)
        assert len(results) >= 1

    def test_top_k_limit(self):
        kb = KnowledgeBase()
        for i in range(10):
            kb.add_text(f"测试文档第{i}篇，关于游戏体验", category="测试")
        results = kb.search("游戏", top_k=3)
        assert len(results) <= 3

    def test_remove_document(self):
        kb = KnowledgeBase()
        kb.add_text("测试文档", category="测试")
        doc_id = kb.documents[0]["id"]
        assert kb.remove(doc_id) is True
        assert len(kb.documents) == 0

    def test_remove_nonexistent(self):
        kb = KnowledgeBase()
        assert kb.remove("nonexistent-id") is False

    def test_clear_all(self):
        kb = KnowledgeBase()
        kb.add_text("文档1", category="测试")
        kb.add_text("文档2", category="测试")
        kb.add_glossary_entry("MMO", "大型多人在线游戏")
        kb.clear()
        assert len(kb.documents) == 0

    def test_stats(self):
        kb = KnowledgeBase()
        kb.add_text("访谈记录1", category="访谈记录")
        kb.add_text("访谈记录2", category="访谈记录")
        kb.add_glossary_entry("RPG", "角色扮演游戏")
        kb.add_glossary_entry("MMO", "大型多人在线游戏")
        stats = kb.get_stats()
        assert stats["total"] == 4
        assert stats["categories"].get("访谈记录", 0) == 2
        assert stats["categories"].get("术语表", 0) == 2

    def test_long_text_chunking(self):
        kb = KnowledgeBase()
        long_text = "词语 " * 1000  # long text that should be chunked
        count = kb.add_text(long_text, category="测试")
        assert count > 1  # should be split into multiple chunks

    def test_category_filtering(self):
        kb = KnowledgeBase()
        kb.add_text("游戏相关", category="访谈记录")
        kb.add_glossary_entry("FPS", "第一人称射击游戏")
        results = kb.search("游戏", categories=["术语表"], top_k=5)
        for r in results:
            assert r["category"] == "术语表"