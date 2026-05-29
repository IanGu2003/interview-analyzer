"""Interview Intelligence Organization & Analysis Agent - Streamlit App

A standalone web application for processing interview audio, matching responses
to structured questions, and generating dual-version reports (raw + coded).
Supports OpenAI-compatible APIs (DeepSeek, OpenAI, etc.)
"""
import os
import sys
import json
import tempfile
import uuid
from pathlib import Path
from datetime import datetime

import streamlit as st

# Add parent to path for utils imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import memory as mem
from utils import asr as asr_utils
from utils import llm_utils
from utils import report

# ---------- Page Config ----------
st.set_page_config(
    page_title="访谈智能整理与分析系统",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Session State ----------
if "processing" not in st.session_state:
    st.session_state.processing = False
if "raw_report_path" not in st.session_state:
    st.session_state.raw_report_path = None
if "coded_report_path" not in st.session_state:
    st.session_state.coded_report_path = None
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "match_summary" not in st.session_state:
    st.session_state.match_summary = {}
if "questions" not in st.session_state:
    st.session_state.questions = []
if "audio_file_name" not in st.session_state:
    st.session_state.audio_file_name = ""

# ---------- Default Questions ----------
DEFAULT_QUESTIONS = [
    {"question_id": "Q1", "question_text": "你平时玩什么类型的游戏？"},
    {"question_id": "Q2", "question_text": "你每天大概花多少时间在游戏上？"},
    {"question_id": "Q3", "question_text": "你最喜欢的一款游戏是什么？为什么？"},
    {"question_id": "Q4", "question_text": "你关注游戏画面的哪些方面？"},
    {"question_id": "Q5", "question_text": "游戏的剧情和叙事对你来说重要吗？"},
    {"question_id": "Q6", "question_text": "你在游戏中更喜欢单人还是多人模式？"},
    {"question_id": "Q7", "question_text": "你对游戏内的付费机制怎么看？"},
    {"question_id": "Q8", "question_text": "你通常在什么平台上玩游戏？"},
    {"question_id": "Q9", "question_text": "你有过因为游戏体验差而弃坑的经历吗？"},
    {"question_id": "Q10", "question_text": "你如何发现和选择新游戏？"},
    {"question_id": "Q11", "question_text": "你觉得一个好的游戏应该具备什么核心要素？"},
    {"question_id": "Q12", "question_text": "你对游戏社交功能（如好友、公会、语音）有什么看法？"},
    {"question_id": "Q13", "question_text": "你玩游戏时更关注胜负还是娱乐体验？"},
    {"question_id": "Q14", "question_text": "游戏更新和运营活动会影响你的留存吗？"},
    {"question_id": "Q15", "question_text": "你会向朋友推荐什么样的游戏？"},
    {"question_id": "Q16", "question_text": "你认为手游和端游在体验上最大的区别是什么？"},
]


# ---------- Sidebar: API Configuration ----------
with st.sidebar:
    st.title("⚙️ 配置")
    st.markdown("---")

    with st.expander("🔑 API 设置", expanded=True):
        api_base = st.text_input(
            "API Base URL",
            value="https://api.deepseek.com/v1",
            help="支持 OpenAI 兼容接口的 API 地址",
            placeholder="https://api.deepseek.com/v1",
        )
        api_key = st.text_input(
            "API Key",
            type="password",
            help="请填写你的 API Key",
            placeholder="sk-...",
        )
        llm_model = st.text_input(
            "LLM 模型",
            value="deepseek-chat",
            help="用于文本分析和编码的大模型",
            placeholder="deepseek-chat",
        )
        whisper_model = st.text_input(
            "Whisper 模型",
            value="whisper-1",
            help="用于语音转写的模型",
            placeholder="whisper-1",
        )

    st.markdown("---")
    st.caption("💡 支持 OpenAI 兼容接口的 API")
    st.caption("已适配：DeepSeek、OpenAI、Azure OpenAI、智谱、百炼等")

    st.markdown("---")
    st.caption("📁 输出：每次处理将自动生成")
    st.caption("1️⃣ **原话版.xlsx** — 问题与回答一一对应")
    st.caption("2️⃣ **初步编码版.xlsx** — 含主题编码/关键词/情感")


# ---------- Main UI ----------
st.title("🎙️ 访谈智能整理与分析系统")
st.markdown("""
上传访谈录音文件和结构化访谈框架，自动完成：
- **语音转写** → 自动识别受访者回答
- **语义匹配** → 将回答匹配到对应问题
- **双版本报告** → 同时输出原话版和初步编码版
""")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📤 上传与处理", "📋 访谈问题编辑", "❓ 使用说明"])

# ===== Tab 1: Upload & Process =====
with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("🎵 上传访谈录音")
        audio_file = st.file_uploader(
            "支持格式：.m4a, .mp3, .wav, .ogg 等",
            type=["m4a", "mp3", "wav", "ogg", "aac", "wma", "flac"],
            help="请上传包含受访者回答的访谈录音文件",
        )

        if audio_file:
            st.audio(audio_file, format=f"audio/{audio_file.name.split('.')[-1]}")
            st.session_state.audio_file_name = audio_file.name

    with col2:
        st.subheader("📋 访谈框架")
        use_default = st.checkbox("使用默认访谈问题（16题）", value=True)

        if not use_default:
            questions_json = st.text_area(
                "手动输入 JSON 格式的访谈问题",
                height=200,
                value=json.dumps(DEFAULT_QUESTIONS, ensure_ascii=False, indent=2),
            )
        else:
            questions_json = json.dumps(DEFAULT_QUESTIONS, ensure_ascii=False, indent=2)
            st.info(f"✅ 已加载 {len(DEFAULT_QUESTIONS)} 道默认问题", icon="📋")

    st.markdown("---")

    # Process button
    can_process = (
        audio_file is not None
        and api_key
        and not st.session_state.processing
    )

    if not can_process:
        if not audio_file:
            st.warning("请先上传录音文件", icon="⚠️")
        elif not api_key:
            st.warning("请在左侧配置 API Key", icon="⚠️")
        elif st.session_state.processing:
            st.warning("正在处理中，请稍候...", icon="⏳")

    if st.button(
        "🚀 开始分析",
        type="primary",
        use_container_width=True,
        disabled=not can_process,
    ):
        # Parse questions
        try:
            questions = json.loads(questions_json)
            st.session_state.questions = questions
        except json.JSONDecodeError as e:
            st.error(f"❌ 访谈问题 JSON 格式错误: {e}")
            st.stop()

        # Start processing
        st.session_state.processing = True
        progress_bar = st.progress(0, text="初始化...")
        status_text = st.empty()

        try:
            # Save uploaded file
            suffix = Path(audio_file.name).suffix
            tmp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(tmp_dir, f"audio{suffix}")
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            progress_bar.progress(5, text="音频已保存...")

            # ---------- Step 1: Audio Preprocessing ----------
            status_text.info("🔄 步骤1/5：音频预处理...")
            progress_bar.progress(10, text="音频预处理...")

            # Convert if not standard WAV
            wav_path = None
            if suffix.lower() != ".wav":
                if asr_utils.check_ffmpeg():
                    wav_path = asr_utils.convert_to_standard_wav(audio_path)
                    if wav_path:
                        status_text.info(f"✅ 音频已转为标准WAV格式")
                    else:
                        st.warning("⚠️ ffmpeg 转码失败，尝试直接使用原始文件")
                        wav_path = audio_path
                else:
                    st.warning("⚠️ 未检测到 ffmpeg，使用原始文件进行转写")
                    wav_path = audio_path
            else:
                wav_path = audio_path

            file_size_mb = asr_utils.get_file_size_mb(wav_path)
            status_text.info(f"📊 音频大小: {file_size_mb:.1f}MB")

            progress_bar.progress(20, text="准备调用语音转写...")

            # ---------- Step 2: ASR ----------
            status_text.info("🔄 步骤2/5：语音转写中（Whisper API）...")
            progress_bar.progress(30, text="语音转写中...")

            asr_result = asr_utils.transcribe_with_whisper_api(
                audio_path=wav_path,
                api_key=api_key,
                base_url=api_base,
                model=whisper_model,
                language="zh",
            )

            transcript = asr_result["full_text"]
            st.session_state.transcript = transcript

            if not transcript.strip():
                st.error("❌ 语音转写结果为空，请检查音频文件是否有效")
                st.stop()

            status_text.success(f"✅ 转写完成：{len(transcript)} 字符")
            progress_bar.progress(50, text="转写完成")

            # ---------- Step 3: LLM Analysis ----------
            status_text.info("🔄 步骤3/5：LLM分析与语义匹配中...")
            progress_bar.progress(60, text="LLM 语义分析中...")

            client = llm_utils.get_client(api_key, base_url=api_base)

            # Extract answers and match to questions
            matches = llm_utils.extract_answers_from_transcript(
                client=client,
                transcript=transcript,
                questions=questions,
                model=llm_model,
            )

            # Store responses in memory
            interview_memory = mem.reset_memory()
            matched_count = 0

            for m in matches:
                qid = m.get("question_id", "")
                cleaned_text = m.get("cleaned_text", "")
                score = m.get("score", 0.5)
                original_text = m.get("original_text", "")

                if qid and cleaned_text.strip():
                    interview_memory.store_response(
                        question_id=qid,
                        text=cleaned_text,
                        confidence=score,
                        original_text=original_text,
                    )
                    matched_count += 1

            # Build summary
            total_questions = len(questions)
            answered_qs = len(interview_memory.get_all_responses())

            st.session_state.match_summary = {
                "total_segments": len(matches),
                "matched_responses": matched_count,
                "answered_questions": answered_qs,
                "total_questions": total_questions,
                "unanswered": total_questions - answered_qs,
            }

            status_text.success(f"✅ 匹配完成：{matched_count} 条回答匹配到 {answered_qs} 个问题")
            progress_bar.progress(80, text="匹配完成")

            # ---------- Step 4: Generate Reports ----------
            status_text.info("🔄 步骤4/5：生成双版本报告...")
            progress_bar.progress(85, text="生成报告（原话版）...")

            output_prefix = os.path.join(tmp_dir, f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            raw_path, coded_path = report.generate_dual_reports(
                questions=questions,
                interview_memory=interview_memory,
                output_prefix=output_prefix,
                client=client,
                model=llm_model,
            )

            # Load files for download
            with open(raw_path, "rb") as f:
                raw_bytes = f.read()
            with open(coded_path, "rb") as f:
                coded_bytes = f.read()

            st.session_state.raw_report_bytes = raw_bytes
            st.session_state.coded_report_bytes = coded_bytes
            st.session_state.raw_report_path = raw_path
            st.session_state.coded_report_path = coded_path

            progress_bar.progress(100, text="✅ 全部完成！")
            status_text.success("🎉 分析完成！")

        except Exception as e:
            st.error(f"❌ 处理过程中发生错误: {e}")
            import traceback
            st.code(traceback.format_exc(), language="python")
        finally:
            st.session_state.processing = False

    # ---------- Results Section ----------
    if st.session_state.get("raw_report_bytes"):
        st.markdown("---")
        st.subheader("📊 分析结果")

        # Summary
        summary = st.session_state.match_summary
        if summary:
            cols = st.columns(4)
            cols[0].metric("总匹配回答", summary.get("matched_responses", 0))
            cols[1].metric("已回答问题", summary.get("answered_questions", 0))
            cols[2].metric("总问题数", summary.get("total_questions", 0))
            cols[3].metric("未获取回答", summary.get("unanswered", 0))

        # Download buttons
        st.markdown("#### 📥 下载报告")
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.download_button(
                label="📄 下载原话版报告 (.xlsx)",
                data=st.session_state.raw_report_bytes,
                file_name=f"访谈报告_原话版_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with dl_col2:
            st.download_button(
                label="📊 下载初步编码版报告 (.xlsx)",
                data=st.session_state.coded_report_bytes,
                file_name=f"访谈报告_初步编码版_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        # Show transcript preview
        if st.session_state.transcript:
            with st.expander("📝 查看转写全文"):
                st.text_area("转写文本", st.session_state.transcript, height=300)


# ===== Tab 2: Edit Questions =====
with tab2:
    st.subheader("📋 编辑结构化访谈问题")
    st.caption("你可以编辑以下默认问题列表，或替换为你自己的访谈框架")

    questions_editor = st.data_editor(
        DEFAULT_QUESTIONS,
        column_config={
            "question_id": st.column_config.TextColumn("问题ID", width="small"),
            "question_text": st.column_config.TextColumn("问题文本", width="large"),
        },
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )

    st.session_state.questions = questions_editor

    if st.button("📥 保存为默认问题"):
        st.success(f"✅ 已保存 {len(questions_editor)} 道问题")


# ===== Tab 3: Usage Guide =====
with tab3:
    st.markdown("""
    ## 📖 使用说明

    ### 快速开始
    1. **左侧配置**：填写 API Base URL 和 API Key（支持 DeepSeek / OpenAI 等）
    2. **上传音频**：上传包含受访者回答的访谈录音（.m4a, .mp3, .wav）
    3. **编辑问题**：调整面试问题框架（支持增删改）
    4. **点击分析**：系统自动完成转写 → 匹配 → 编码 → 生成报告

    ### 双版本报告
    | 版本 | 内容 |
    |------|------|
    | **原话版** | 所有问题与受访者回答一一对应，未获取回答的问题黄色标注 |
    | **初步编码版** | 在原话基础上增加：主题编码、子编码、关键词、情感倾向 |

    ### 支持的API
    - **DeepSeek**: `https://api.deepseek.com/v1`
    - **OpenAI**: `https://api.openai.com/v1`
    - **Azure OpenAI**: 自定义 endpoint
    - **智谱开放平台**: `https://open.bigmodel.cn/api/paas/v4`
    - **阿里云百炼**: 自定义 endpoint

    ### 注意事项
    - ⚠️ 单声道录音中主持人与受访者混合，系统会自动识别提取受访者回答
    - 🎯 保持问题文本清晰具体，有助于提高语义匹配准确率
    - 📂 报告为 .xlsx 格式，可用 Excel 或 WPS 打开
    """)

# ---------- Footer ----------
st.markdown("---")
st.caption("🎙️ 访谈智能整理与分析系统 v1.0 | 基于 LLM + Whisper 构建")