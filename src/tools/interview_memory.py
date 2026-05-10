"""
响应存储工具 - 存储匹配成功的回答到记忆模块
"""
import json
from typing import Dict, List, Any
from langchain.tools import tool
from datetime import datetime


class InterviewMemory:
    """访谈记忆模块 - 存储问题和回答的映射"""
    
    def __init__(self):
        self.responses: Dict[str, List[Dict]] = {}  # question_id -> [responses]
        self.unmatched: List[Dict] = []  # 未匹配的片段
        self.fuzzy_match: List[Dict] = []  # 模糊匹配的片段（待人工确认）
        self.stats = {
            "total_segments": 0,
            "relevant_segments": 0,
            "irrelevant_segments": 0,
            "stored_responses": 0,
            "unmatched_count": 0,
            "fuzzy_match_count": 0
        }
    
    def store_response(self, question_id: str, text: str, 
                      timestamp: str = None, confidence: float = 1.0,
                      original_text: str = None) -> str:
        """
        存储回答到指定问题下
        
        Args:
            question_id: 问题ID
            text: 清理后的回答文本
            timestamp: 时间戳
            confidence: 置信度
            original_text: 原始文本（如果有语气词清理）
        
        Returns:
            存储结果
        """
        if question_id not in self.responses:
            self.responses[question_id] = []
        
        response_entry = {
            "text": text,
            "timestamp": timestamp,
            "confidence": confidence,
            "original_text": original_text,
            "stored_at": datetime.now().isoformat()
        }
        
        self.responses[question_id].append(response_entry)
        self.stats["stored_responses"] += 1
        
        return json.dumps({"status": "stored", "question_id": question_id}, ensure_ascii=False)
    
    def add_unmatched(self, text: str, timestamp: str = None, 
                     reason: str = "无匹配问题") -> str:
        """添加未匹配的片段"""
        self.unmatched.append({
            "text": text,
            "timestamp": timestamp,
            "reason": reason,
            "stored_at": datetime.now().isoformat()
        })
        self.stats["unmatched_count"] += 1
        return json.dumps({"status": "unmatched"}, ensure_ascii=False)
    
    def add_fuzzy_match(self, text: str, matched_question_id: str,
                       score: float, timestamp: str = None) -> str:
        """添加模糊匹配的片段（需要人工确认）"""
        self.fuzzy_match.append({
            "text": text,
            "matched_question_id": matched_question_id,
            "score": score,
            "timestamp": timestamp,
            "needs_verification": True,
            "stored_at": datetime.now().isoformat()
        })
        self.stats["fuzzy_match_count"] += 1
        return json.dumps({"status": "fuzzy_match", "question_id": matched_question_id}, ensure_ascii=False)
    
    def update_stats(self, relevant: bool = True):
        """更新统计信息"""
        self.stats["total_segments"] += 1
        if relevant:
            self.stats["relevant_segments"] += 1
        else:
            self.stats["irrelevant_segments"] += 1
    
    def get_all_responses(self) -> Dict[str, List[Dict]]:
        """获取所有回答"""
        return self.responses
    
    def get_responses_by_question(self, question_id: str) -> List[Dict]:
        """获取指定问题的所有回答"""
        return self.responses.get(question_id, [])
    
    def get_full_memory(self) -> Dict[str, Any]:
        """获取完整的记忆数据"""
        return {
            "responses": self.responses,
            "unmatched": self.unmatched,
            "fuzzy_match": self.fuzzy_match,
            "stats": self.stats
        }
    
    def export_to_json(self) -> str:
        """导出为JSON"""
        return json.dumps(self.get_full_memory(), ensure_ascii=False, indent=2)


# 全局记忆实例
_global_memory = None


def get_memory() -> InterviewMemory:
    """获取全局记忆实例"""
    global _global_memory
    if _global_memory is None:
        _global_memory = InterviewMemory()
    return _global_memory


def reset_memory():
    """重置全局记忆"""
    global _global_memory
    _global_memory = InterviewMemory()


@tool
def store_response(question_id: str, text: str, timestamp: str = None,
                  confidence: float = 1.0, original_text: str = None) -> str:
    """
    存储匹配成功的回答
    
    Args:
        question_id: 问题ID
        text: 清理后的回答文本
        timestamp: 时间戳（格式：HH:MM:SS）
        confidence: 置信度（0.0-1.0）
        original_text: 原始文本
    
    Returns:
        存储结果
    """
    memory = get_memory()
    return memory.store_response(question_id, text, timestamp, confidence, original_text)


@tool
def store_unmatched(text: str, timestamp: str = None, reason: str = "无匹配问题") -> str:
    """
    存储未匹配的片段
    
    Args:
        text: 文本片段
        timestamp: 时间戳
        reason: 未匹配原因
    
    Returns:
        存储结果
    """
    memory = get_memory()
    return memory.add_unmatched(text, timestamp, reason)


@tool
def store_fuzzy_match(text: str, matched_question_id: str, 
                     score: float, timestamp: str = None) -> str:
    """
    存储模糊匹配的片段（需要人工确认）
    
    Args:
        text: 文本片段
        matched_question_id: 匹配到的问题ID
        score: 匹配分数
        timestamp: 时间戳
    
    Returns:
        存储结果
    """
    memory = get_memory()
    return memory.add_fuzzy_match(text, matched_question_id, score, timestamp)


@tool
def get_memory_summary() -> str:
    """
    获取记忆摘要
    
    Returns:
        记忆摘要JSON
    """
    memory = get_memory()
    summary = {
        "total_questions": len(memory.responses),
        "total_responses": memory.stats["stored_responses"],
        "unmatched_count": memory.stats["unmatched_count"],
        "fuzzy_match_count": memory.stats["fuzzy_match_count"],
        "stats": memory.stats
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


@tool
def export_interview_data() -> str:
    """
    导出所有访谈数据
    
    Returns:
        完整的访谈数据JSON
    """
    memory = get_memory()
    return memory.export_to_json()
