"""
后处理输出工具 - 生成最终报告文件（Excel/Word/Markdown）
"""
import json
from typing import Dict, List, Any
from langchain.tools import tool
from datetime import datetime
from tools.interview_memory import get_memory


@tool
def postprocess_output(output_format: str = "excel", questions: str = None) -> str:
    """
    将访谈数据格式化为最终报告
    
    Args:
        output_format: 输出格式，可选 "excel", "word", "markdown", "json"
        questions: 问题列表JSON（用于补充问题文本）
    
    Returns:
        输出文件的路径或内容
    """
    memory = get_memory()
    questions_data = json.loads(questions) if questions else None
    
    if output_format == "json":
        return memory.export_to_json()
    elif output_format == "excel":
        return _generate_excel(memory, questions_data)
    elif output_format == "word":
        return _generate_word(memory, questions_data)
    elif output_format == "markdown":
        return _generate_markdown(memory, questions_data)
    else:
        return json.dumps({"error": f"不支持的格式: {output_format}"})


def _generate_excel(memory, questions_data: List[Dict] = None) -> str:
    """生成Excel文件"""
    import os
    
    # 构建问题ID到文本的映射
    q_map = {}
    if questions_data:
        for q in questions_data:
            q_id = q.get("question_id", "")
            q_text = q.get("question_text", "")
            q_map[q_id] = q_text
    
    # 准备数据
    rows = []
    for q_id, responses in memory.responses.items():
        q_text = q_map.get(q_id, q_id)
        for resp in responses:
            rows.append({
                "问题ID": q_id,
                "问题文本": q_text,
                "受访者原话": resp.get("text", ""),
                "时间戳": resp.get("timestamp", ""),
                "置信度": resp.get("confidence", 1.0)
            })
    
    # 如果没有数据，返回提示
    if not rows:
        return json.dumps({"message": "暂无匹配数据", "data": []})
    
    # 保存为CSV（Excel兼容）
    output_path = f"/tmp/interview_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    try:
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=["问题ID", "问题文本", "受访者原话", "时间戳", "置信度"])
            writer.writeheader()
            writer.writerows(rows)
        
        return json.dumps({
            "status": "success",
            "format": "csv",
            "path": output_path,
            "rows_count": len(rows)
        })
    except Exception as e:
        return json.dumps({"error": f"生成CSV失败: {str(e)}"})


def _generate_word(memory, questions_data: List[Dict] = None) -> str:
    """生成Word文档"""
    # 构建问题ID到文本的映射
    q_map = {}
    if questions_data:
        for q in questions_data:
            q_id = q.get("question_id", "")
            q_text = q.get("question_text", "")
            q_map[q_id] = q_text
    
    # 生成Markdown内容（Word可以转换为Markdown）
    content = _generate_markdown_content(memory, q_map)
    
    # 保存为markdown文件
    output_path = f"/tmp/interview_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return json.dumps({
            "status": "success",
            "format": "markdown",
            "path": output_path,
            "content_preview": content[:500]
        })
    except Exception as e:
        return json.dumps({"error": f"生成文件失败: {str(e)}"})


def _generate_markdown(memory, questions_data: List[Dict] = None) -> str:
    """生成Markdown内容"""
    q_map = {}
    if questions_data:
        for q in questions_data:
            q_id = q.get("question_id", "")
            q_text = q.get("question_text", "")
            q_map[q_id] = q_text
    
    content = _generate_markdown_content(memory, q_map)
    
    # 保存文件
    output_path = f"/tmp/interview_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return json.dumps({
            "status": "success",
            "format": "markdown",
            "path": output_path,
            "content": content
        })
    except Exception as e:
        return json.dumps({"error": f"生成Markdown失败: {str(e)}"})


def _generate_markdown_content(memory, q_map: Dict) -> str:
    """生成Markdown格式的内容"""
    lines = []
    lines.append(f"# 访谈结果整理\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"---\n\n")
    
    # 统计信息
    lines.append(f"## 处理统计\n\n")
    lines.append(f"- 总片段数: {memory.stats['total_segments']}\n")
    lines.append(f"- 有效片段数: {memory.stats['relevant_segments']}\n")
    lines.append(f"- 丢弃片段数: {memory.stats['irrelevant_segments']}\n")
    lines.append(f"- 存储回答数: {memory.stats['stored_responses']}\n")
    lines.append(f"- 未匹配片段数: {memory.stats['unmatched_count']}\n")
    lines.append(f"- 模糊匹配片段数: {memory.stats['fuzzy_match_count']}\n")
    lines.append(f"\n---\n\n")
    
    # 按问题分组展示
    lines.append(f"## 回答详情\n\n")
    
    for q_id in sorted(memory.responses.keys()):
        responses = memory.responses[q_id]
        q_text = q_map.get(q_id, q_id)
        
        lines.append(f"### {q_id}: {q_text}\n\n")
        lines.append(f"| # | 受访者原话 | 时间戳 | 置信度 |\n")
        lines.append(f"|---|-----------|--------|--------|\n")
        
        for i, resp in enumerate(responses, 1):
            text = resp.get("text", "").replace("|", "\\|").replace("\n", " ")
            timestamp = resp.get("timestamp", "-")
            confidence = f"{resp.get('confidence', 1.0):.2f}"
            lines.append(f"| {i} | {text} | {timestamp} | {confidence} |\n")
        
        lines.append(f"\n")
    
    # 未匹配片段
    if memory.unmatched:
        lines.append(f"---\n\n")
        lines.append(f"## 未匹配片段\n\n")
        for i, item in enumerate(memory.unmatched, 1):
            text = item.get("text", "").replace("|", "\\|")
            timestamp = item.get("timestamp", "-")
            reason = item.get("reason", "")
            lines.append(f"{i}. {text} {f'(时间: {timestamp})' if timestamp != '-' else ''} - {reason}\n")
        lines.append(f"\n")
    
    # 模糊匹配片段
    if memory.fuzzy_match:
        lines.append(f"---\n\n")
        lines.append(f"## 待人工确认的模糊匹配\n\n")
        for i, item in enumerate(memory.fuzzy_match, 1):
            text = item.get("text", "").replace("|", "\\|")
            matched_q = item.get("matched_question_id", "")
            score = item.get("score", 0)
            timestamp = item.get("timestamp", "-")
            lines.append(f"{i}. **{text}**\n")
            lines.append(f"   - 匹配问题: {matched_q}, 得分: {score:.2f}, 时间: {timestamp}\n")
        lines.append(f"\n")
    
    return "".join(lines)


def generate_processing_report() -> str:
    """
    生成处理报告摘要
    
    Returns:
        处理报告JSON
    """
    memory = get_memory()
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "statistics": memory.stats,
        "questions_answered": list(memory.responses.keys()),
        "unmatched_count": len(memory.unmatched),
        "fuzzy_match_count": len(memory.fuzzy_match)
    }
    
    return json.dumps(report, ensure_ascii=False, indent=2)
