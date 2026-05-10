"""
访谈智能整理与分析Agent - 基于LangGraph的主控Agent
"""
import os
import json
from typing import Annotated, List, Dict, Any, Optional
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from coze_coding_utils.runtime_ctx.context import default_headers

# 导入工具
from tools.speech_to_text import speech_to_text
from tools.text_cleaner import text_cleaner
from tools.segment_text import segment_with_timestamps, ms_to_time
from tools.relevance_classifier import relevance_classifier
from tools.question_matcher import question_matcher
from tools.interview_memory import (
    get_memory, reset_memory, store_response, 
    store_unmatched, store_fuzzy_match, get_memory_summary
)
from tools.postprocess_output import postprocess_output, generate_processing_report

LLM_CONFIG = "config/agent_llm_config.json"

# 默认保留最近 20 轮对话 (40 条消息)
MAX_MESSAGES = 40

# 匹配阈值配置
HIGH_CONFIDENCE_THRESHOLD = 0.7
LOW_CONFIDENCE_THRESHOLD = 0.4


class AgentState(MessagesState):
    """Agent状态定义"""
    pass


def build_agent(ctx=None):
    """
    构建访谈智能整理与分析Agent
    
    Returns:
        配置好的Agent实例
    """
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, LLM_CONFIG)

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
    base_url = os.getenv("COZE_INTEGRATION_MODEL_BASE_URL")

    llm = ChatOpenAI(
        model=cfg['config'].get("model"),
        api_key=api_key,
        base_url=base_url,
        temperature=cfg['config'].get('temperature', 0.7),
        streaming=True,
        timeout=cfg['config'].get('timeout', 600),
        extra_body={
            "thinking": {
                "type": cfg['config'].get('thinking', 'disabled')
            }
        },
        default_headers=default_headers(ctx) if ctx else {}
    )

    # 获取工具列表
    tools = [
        speech_to_text,
        text_cleaner,
        segment_with_timestamps,
        relevance_classifier,
        question_matcher,
        store_response,
        store_unmatched,
        store_fuzzy_match,
        get_memory_summary,
        postprocess_output
    ]

    return create_agent(
        model=llm,
        system_prompt=cfg.get("sp"),
        tools=tools,
        checkpointer=None,
        state_schema=AgentState,
    )


def create_interview_workflow():
    """
    创建访谈处理工作流（LangGraph StateGraph）
    
    这是一个可选的高级实现，提供更精细的流程控制
    """
    from typing import TypedDict
    
    # 定义状态
    class WorkflowState(TypedDict):
        audio_path: str
        questions: List[Dict]
        transcribed_text: str
        segments: List[Dict]
        current_index: int
        memory: Dict
        final_output: Optional[str]
    
    # 节点定义
    def transcribe_node(state: WorkflowState) -> WorkflowState:
        """转写音频"""
        result = speech_to_text.invoke({"audio_path": state["audio_path"]})
        data = json.loads(result)
        
        return {
            **state,
            "transcribed_text": data.get("full_text", ""),
            "segments": data.get("segments", []),
            "current_index": 0
        }
    
    def process_segment_node(state: WorkflowState) -> WorkflowState:
        """处理单个片段"""
        segments = state["segments"]
        current_idx = state["current_index"]
        
        if current_idx >= len(segments):
            return {**state, "memory": get_memory().get_full_memory()}
        
        segment = segments[current_idx]
        text = segment.get("text", "")
        timestamp = segment.get("start_time", 0)
        
        # 格式化时间戳
        time_str = ms_to_time(timestamp)
        
        # 相关性判断
        relevance_result = relevance_classifier.invoke({
            "text": text,
            "questions": json.dumps(state["questions"])
        })
        relevance_data = json.loads(relevance_result)
        relevance_score = relevance_data.get("score", 0)
        
        if relevance_score < LOW_CONFIDENCE_THRESHOLD:
            store_unmatched.invoke({
                "text": text,
                "timestamp": time_str,
                "reason": f"相关性过低 ({relevance_score:.2f})"
            })
            return {**state, "current_index": current_idx + 1}
        
        # 清理语气词
        cleaned_text = text_cleaner.invoke({"text": text})
        
        # 问题匹配
        match_result = question_matcher.invoke({
            "text": cleaned_text,
            "questions": json.dumps(state["questions"])
        })
        match_data = json.loads(match_result)
        
        q_id = match_data.get("question_id", "unmatched")
        match_score = match_data.get("score", 0)
        
        if q_id == "unmatched":
            store_unmatched.invoke({
                "text": cleaned_text,
                "timestamp": time_str,
                "reason": "无匹配问题"
            })
        elif match_score >= HIGH_CONFIDENCE_THRESHOLD:
            store_response.invoke({
                "question_id": q_id,
                "text": cleaned_text,
                "timestamp": time_str,
                "confidence": match_score,
                "original_text": text
            })
        else:
            store_fuzzy_match.invoke({
                "text": cleaned_text,
                "matched_question_id": q_id,
                "score": match_score,
                "timestamp": time_str
            })
        
        return {**state, "current_index": current_idx + 1}
    
    def should_continue(state: WorkflowState) -> bool:
        """判断是否继续处理"""
        return state["current_index"] < len(state["segments"])
    
    def output_node(state: WorkflowState) -> WorkflowState:
        """生成输出"""
        output = postprocess_output.invoke({
            "output_format": "excel",
            "questions": json.dumps(state["questions"])
        })
        return {**state, "final_output": output}
    
    # 构建图
    workflow = StateGraph(WorkflowState)
    
    workflow.add_node("transcribe", transcribe_node)
    workflow.add_node("process_segment", process_segment_node)
    workflow.add_node("output", output_node)
    
    workflow.set_entry_point("transcribe")
    
    workflow.add_conditional_edges(
        "process_segment",
        should_continue,
        {True: "process_segment", False: "output"}
    )
    
    workflow.add_edge("transcribe", "process_segment")
    workflow.add_edge("output", END)
    
    return workflow.compile()


def reset_interview_memory():
    """重置访谈记忆"""
    reset_memory()


def get_interview_summary() -> Dict[str, Any]:
    """获取访谈处理摘要"""
    memory = get_memory()
    return memory.get_full_memory()
