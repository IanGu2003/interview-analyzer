"""Knowledge Base for interview analysis - RAG without external vector DB.

Stores reference documents (past interviews, glossaries, terminology) and
retrieves relevant context to enhance LLM accuracy during:
- Semantic matching (matching responses to questions)
- Thematic coding (maintaining coding consistency)
- Terminology recognition (domain-specific terms)

Design: pure Python keyword + LLM rerank, no external vector DB dependency.
"""
import re
import uuid
import json
from typing import Optional


class KnowledgeBase:
    """Lightweight knowledge base with keyword retrieval"""

    def __init__(self):
        self.documents: list[dict] = []  # [{id, text, category, title, source}]

    # ── Document Management ──────────────────────────────────────

    def add_text(self, text: str, category: str = "general",
                 title: str = "", source: str = "") -> int:
        """Add a text document (will be auto-chunked if long)"""
        chunks = self._chunk_text(text, category=category)
        for chunk in chunks:
            self.documents.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "category": category,
                "title": title,
                "source": source,
            })
        return len(chunks)

    def add_glossary_entry(self, term: str, definition: str,
                           category: str = "术语表") -> None:
        """Add a single glossary term-definition pair"""
        text = f"{term}：{definition}"
        self.documents.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "category": category,
            "title": term,
            "source": "术语表",
        })

    def add_interview_case(self, question: str, answer: str,
                           coding: Optional[dict] = None) -> None:
        """Add a past interview case (question + answer + coding)"""
        text = f"【访谈案例】\n问题：{question}\n回答：{answer}"
        if coding:
            text += f"\n编码：主题({coding.get('theme_code', '')}) / 子编码({coding.get('sub_code', '')}) / 情感({coding.get('sentiment', '')})"
            if coding.get('keywords'):
                text += f" / 关键词({', '.join(coding['keywords'])})"
        self.documents.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "category": "访谈案例",
            "title": question[:40],
            "source": "历史访谈",
        })

    def clear(self) -> None:
        """Clear all documents"""
        self.documents = []

    def remove_by_category(self, category: str) -> int:
        """Remove all documents in a category, returns count removed"""
        before = len(self.documents)
        self.documents = [d for d in self.documents if d["category"] != category]
        return before - len(self.documents)

    # ── Retrieval ────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5,
               category_filter: Optional[list[str]] = None) -> list[dict]:
        """Search documents by keyword relevance scoring"""
        keywords = self._extract_keywords(query)

        filtered = self.documents
        if category_filter:
            filtered = [d for d in filtered if d["category"] in category_filter]

        scored = []
        for doc in filtered:
            score = self._keyword_score(doc["text"], keywords)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:top_k]]

    def get_context(self, query: str, top_k: int = 5,
                    max_chars: int = 3000) -> str:
        """Get formatted context string for LLM prompt injection"""
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        parts = []
        for r in results:
            tag = f"[{r['category']}]"
            if r['title']:
                tag += f" {r['title']}"
            parts.append(f"{tag}\n{r['text']}")

        context = "\n\n---\n".join(parts)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n...（截断）"

        return context

    def count(self) -> dict:
        """Get statistics about knowledge base contents"""
        categories = {}
        for d in self.documents:
            cat = d["category"]
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total": len(self.documents),
            "categories": categories,
        }

    # ── Internals ────────────────────────────────────────────────

    def _chunk_text(self, text: str, category: str = "",
                    chunk_size: int = 600, overlap: int = 50) -> list[str]:
        """Split long text into overlapping chunks"""
        text = text.strip()
        if not text:
            return []

        if len(text) <= chunk_size:
            return [text]

        # Split by paragraph first
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) < chunk_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                current = para

        if current:
            chunks.append(current)

        return chunks if chunks else [text]

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from query text"""
        # Remove common stop words
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不',
                      '人', '都', '一', '一个', '上', '也', '很', '到', '说',
                      '要', '去', '你', '会', '着', '没有', '看', '好', '自己',
                      '这', '他', '她', '它', '们', '那', '什么', '怎么', '为什么',
                      '吗', '啊', '呢', '吧', '嗯', '哦', '呃', '嘛'}

        # Tokenize: take 2-4 character n-grams as keywords
        tokens = set()
        text_clean = re.sub(r'[^\u4e00-\u9fff\w]', ' ', text)
        
        # Add single meaningful Chinese characters (exclude stop words)
        for char in text_clean:
            if '\u4e00' <= char <= '\u9fff' and char not in stop_words:
                tokens.add(char)

        # Add 2-grams
        chars = [c for c in text_clean if '\u4e00' <= c <= '\u9fff' or c.isalpha()]
        for i in range(len(chars) - 1):
            bigram = chars[i] + chars[i + 1]
            if not all(c in stop_words for c in bigram):
                tokens.add(bigram)

        # Add 3-grams
        for i in range(len(chars) - 2):
            trigram = chars[i] + chars[i + 1] + chars[i + 2]
            tokens.add(trigram)

        # Sort by length (longer = more specific) and return
        return sorted(tokens, key=len, reverse=True)[:30]

    def _keyword_score(self, text: str, keywords: list[str]) -> float:
        """Score a document's relevance to keywords"""
        if not keywords:
            return 0.0

        text_lower = text.lower()
        score = 0.0
        matched = 0

        for kw in keywords:
            count = text_lower.count(kw.lower())
            if count > 0:
                score += count * len(kw)  # Longer keywords = higher weight
                matched += 1

        # Normalize by document length
        doc_len = max(len(text), 1)
        score = score / doc_len * 100

        # Bonus for matching many different keywords
        if matched > 1:
            score *= (1 + 0.1 * matched)

        return round(score, 4)


# ── Global Singleton ────────────────────────────────────────────

_kb_instance: Optional[KnowledgeBase] = None


def get_kb() -> KnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance


def reset_kb() -> KnowledgeBase:
    global _kb_instance
    _kb_instance = KnowledgeBase()
    return _kb_instance