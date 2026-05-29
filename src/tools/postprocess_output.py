"""
后处理输出工具 - 生成最终报告文件（Excel/Word/Markdown）
"""
import json
from typing import Dict, List, Any
from langchain.tools import tool
from datetime import datetime
from tools.interview_memory import get_memory


@tool
def postprocess_output(output_format: str = "excel", questions: str = None, mode: str = "raw") -> str:
    """
    将访谈数据格式化为最终报告
    
    Args:
        output_format: 输出格式，可选 "excel", "word", "markdown", "json"
        questions: 问题列表JSON（用于补充问题文本）
        mode: 输出模式，可选 "raw"（原话版）, "coded"（编码版）, "both"（同时输出两个版本）
    
    Returns:
        输出文件的路径或可下载URL
    """
    memory = get_memory()
    questions_data = json.loads(questions) if questions else None
    
    if output_format == "json":
        return memory.export_to_json()
    elif output_format == "excel":
        if mode == "raw":
            return _generate_raw_excel(memory, questions_data)
        elif mode == "coded":
            return _generate_coded_excel(memory, questions_data)
        elif mode == "both":
            raw_result = _generate_raw_excel(memory, questions_data)
            coded_result = _generate_coded_excel(memory, questions_data)
            try:
                raw_data = json.loads(raw_result)
                coded_data = json.loads(coded_result)
                return json.dumps({
                    "status": "success",
                    "raw": raw_data,
                    "coded": coded_data,
                    "message": "已同时生成原话版和编码版两个报告"
                }, ensure_ascii=False)
            except:
                return json.dumps({"error": "生成双版本报告失败"}, ensure_ascii=False)
        else:
            return json.dumps({"error": f"不支持的输出模式: {mode}"}, ensure_ascii=False)
    elif output_format == "word":
        return _generate_word(memory, questions_data)
    elif output_format == "markdown":
        return _generate_markdown(memory, questions_data)
    else:
        return json.dumps({"error": f"不支持的格式: {output_format}"})


def _generate_raw_excel(memory, questions_data: List[Dict] = None) -> str:
    """
    生成【原话版】Excel - 所有问题一一对应列出，有回答的填回答，无回答的标注
    
    Args:
        memory: InterviewMemory实例
        questions_data: 完整问题列表（用于列出所有问题）
    Returns:
        JSON，包含下载URL
    """
    import os
    import csv
    from coze_coding_dev_sdk.s3 import S3SyncStorage
    
    q_map = {}
    if questions_data:
        for q in questions_data:
            q_id = q.get("question_id", "")
            q_text = q.get("question_text", "")
            q_map[q_id] = q_text
    
    # 如果没传完整问题列表，则用记忆中的问题
    all_qids = list(q_map.keys()) if q_map else sorted(memory.responses.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    
    rows = []
    for q_id in all_qids:
        q_text = q_map.get(q_id, q_id)
        responses = memory.get_responses_by_question(q_id)
        if responses:
            for resp in responses:
                rows.append({
                    "问题ID": q_id,
                    "问题文本": q_text,
                    "受访者原话": resp.get("text", ""),
                    "置信度": resp.get("confidence", 1.0),
                    "说明": "✅ 已匹配"
                })
        else:
            rows.append({
                "问题ID": q_id,
                "问题文本": q_text,
                "受访者原话": "",
                "置信度": "",
                "说明": "⚠️ 未获得直接回答（单声道录音限制）"
            })
    
    if not rows:
        return json.dumps({"message": "暂无数据", "data": []})
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"/tmp/interview_raw_{timestamp}.csv"
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=["问题ID", "问题文本", "受访者原话", "置信度", "说明"])
            writer.writeheader()
            writer.writerows(rows)
        
        storage = S3SyncStorage(
            endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
            access_key="",
            secret_key="",
            bucket_name=os.getenv("COZE_BUCKET_NAME"),
            region="cn-beijing",
        )
        
        with open(output_path, 'rb') as f:
            file_content = f.read()
        
        file_key = storage.upload_file(
            file_content=file_content,
            file_name=f"interview_raw_{timestamp}.csv",
            content_type="text/csv; charset=utf-8",
        )
        
        download_url = storage.generate_presigned_url(key=file_key, expire_time=86400)
        
        return json.dumps({
            "status": "success",
            "version": "raw",
            "format": "csv",
            "description": "原话版：所有问题一一对应，受访者原话如实记录",
            "path": output_path,
            "download_url": download_url,
            "rows_count": len(rows)
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"生成原话版CSV失败: {str(e)}"})


def _generate_coded_excel(memory, questions_data: List[Dict] = None) -> str:
    """
    生成【编码版】Excel - 在原话基础上进行初步编码分析（主题编码、关键词、情感倾向）
    """
    import os
    import csv
    from coze_coding_dev_sdk.s3 import S3SyncStorage
    from coze_coding_dev_sdk import LLMClient
    from coze_coding_utils.runtime_ctx.context import new_context
    from langchain_core.messages import SystemMessage, HumanMessage
    
    q_map = {}
    if questions_data:
        for q in questions_data:
            q_id = q.get("question_id", "")
            q_text = q.get("question_text", "")
            q_map[q_id] = q_text
    
    all_qids = list(q_map.keys()) if q_map else sorted(memory.responses.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    
    # 收集需要编码的回答
    raw_responses = []
    for q_id in all_qids:
        responses = memory.get_responses_by_question(q_id)
        if responses:
            for resp in responses:
                raw_responses.append({
                    "question_id": q_id,
                    "question_text": q_map.get(q_id, q_id),
                    "text": resp.get("text", "")
                })
    
    # 使用LLM进行编码分析
    coded_rows = []
    if raw_responses:
        try:
            ctx = new_context(method="interview_coding")
            client = LLMClient(ctx=ctx)
            
            CODING_SYSTEM = '''你是一个质性研究编码助手。对每个受访者回答进行初步编码分析。

输出JSON数组格式：
[
  {
    "question_id": "Q5",
    "theme_code": "满意度评价",
    "sub_code": "评分反馈",
    "keywords": ["3分", "中等"],
    "sentiment": "中性",
    "coding_notes": "受访者给出中等评分"
  }
]

主题编码（theme_code）规则：
- 产品功能评价：功能满意度、功能缺陷、功能建议
- 使用行为：使用频率、使用场景、使用习惯
- 情感态度：情感连接、推荐意愿、品牌认知
- 竞品对比：竞品功能、竞品体验、迁移原因
- 改进建议：优化方向、新增功能、体验改善

情感倾向（sentiment）可选：正面 / 负面 / 中性 / 混合
'''
            user_prompt = f"请对以下{len(raw_responses)}条受访者回答进行编码分析：\n" + "\n".join(
                [f"{r['question_id']}: {r['question_text']} → 回答：{r['text']}" for r in raw_responses]
            )
            
            msgs = [
                SystemMessage(content=CODING_SYSTEM),
                HumanMessage(content=user_prompt)
            ]
            
            resp = client.invoke(messages=msgs, temperature=0.1)
            
            import re, json as json_mod
            try:
                coding_results = json_mod.loads(resp.content)
            except:
                m = re.search(r'\[[\s\S]*\]', resp.content, re.DOTALL)
                if m:
                    coding_results = json_mod.loads(m.group())
                else:
                    coding_results = []
            
            # 合并编码结果到数据行
            code_map = {}
            for cr in coding_results:
                cqid = cr.get("question_id", "")
                code_map[cqid] = cr
            
            for q_id in all_qids:
                responses = memory.get_responses_by_question(q_id)
                q_text = q_map.get(q_id, q_id)
                if responses:
                    for resp in responses:
                        text = resp.get("text", "")
                        cr = code_map.get(q_id, {})
                        coded_rows.append({
                            "问题ID": q_id,
                            "问题文本": q_text,
                            "受访者原话": text,
                            "主题编码": cr.get("theme_code", ""),
                            "子编码": cr.get("sub_code", ""),
                            "关键词": ", ".join(cr.get("keywords", [])) if isinstance(cr.get("keywords"), list) else cr.get("keywords", ""),
                            "情感倾向": cr.get("sentiment", ""),
                            "编码备注": cr.get("coding_notes", "")
                        })
                else:
                    coded_rows.append({
                        "问题ID": q_id,
                        "问题文本": q_text,
                        "受访者原话": "",
                        "主题编码": "",
                        "子编码": "",
                        "关键词": "",
                        "情感倾向": "",
                        "编码备注": "⚠️ 未获得直接回答"
                    })
        except Exception as e:
            # LLM编码失败时，降级为仅有原话+空编码
            for q_id in all_qids:
                responses = memory.get_responses_by_question(q_id)
                q_text = q_map.get(q_id, q_id)
                if responses:
                    for resp in responses:
                        coded_rows.append({
                            "问题ID": q_id,
                            "问题文本": q_text,
                            "受访者原话": resp.get("text", ""),
                            "主题编码": "",
                            "子编码": "",
                            "关键词": "",
                            "情感倾向": "",
                            "编码备注": f"⚠️ LLM编码失败: {str(e)[:50]}"
                        })
                else:
                    coded_rows.append({
                        "问题ID": q_id,
                        "问题文本": q_text,
                        "受访者原话": "",
                        "主题编码": "",
                        "子编码": "",
                        "关键词": "",
                        "情感倾向": "",
                        "编码备注": "⚠️ 未获得直接回答"
                    })
    else:
        for q_id in all_qids:
            coded_rows.append({
                "问题ID": q_id,
                "问题文本": q_map.get(q_id, q_id),
                "受访者原话": "",
                "主题编码": "",
                "子编码": "",
                "关键词": "",
                "情感倾向": "",
                "编码备注": "⚠️ 未获得直接回答"
            })
    
    if not coded_rows:
        return json.dumps({"message": "暂无数据", "data": []})
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"/tmp/interview_coded_{timestamp}.csv"
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=["问题ID", "问题文本", "受访者原话", "主题编码", "子编码", "关键词", "情感倾向", "编码备注"])
            writer.writeheader()
            writer.writerows(coded_rows)
        
        storage = S3SyncStorage(
            endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
            access_key="",
            secret_key="",
            bucket_name=os.getenv("COZE_BUCKET_NAME"),
            region="cn-beijing",
        )
        
        with open(output_path, 'rb') as f:
            file_content = f.read()
        
        file_key = storage.upload_file(
            file_content=file_content,
            file_name=f"interview_coded_{timestamp}.csv",
            content_type="text/csv; charset=utf-8",
        )
        
        download_url = storage.generate_presigned_url(key=file_key, expire_time=86400)
        
        return json.dumps({
            "status": "success",
            "version": "coded",
            "format": "csv",
            "description": "编码版：在原话基础上进行主题编码、关键词提取和情感分析",
            "path": output_path,
            "download_url": download_url,
            "rows_count": len(coded_rows)
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"生成编码版CSV失败: {str(e)}"})


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
    """生成Markdown内容并上传对象存储"""
    import os
    from coze_coding_dev_sdk.s3 import S3SyncStorage
    
    q_map = {}
    if questions_data:
        for q in questions_data:
            q_id = q.get("question_id", "")
            q_text = q.get("question_text", "")
            q_map[q_id] = q_text
    
    content = _generate_markdown_content(memory, q_map)
    
    # 保存到/tmp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"/tmp/interview_results_{timestamp}.md"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 上传到对象存储
        storage = S3SyncStorage(
            endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
            access_key="",
            secret_key="",
            bucket_name=os.getenv("COZE_BUCKET_NAME"),
            region="cn-beijing",
        )
        
        with open(output_path, 'rb') as f:
            file_content = f.read()
        
        file_key = storage.upload_file(
            file_content=file_content,
            file_name=f"interview_report_{timestamp}.md",
            content_type="text/markdown",
        )
        download_url = storage.generate_presigned_url(key=file_key, expire_time=86400)
        
        return json.dumps({
            "status": "success",
            "format": "markdown",
            "path": output_path,
            "download_url": download_url,
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
