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


def _get_default(secret_key: str, env_key: str, fallback: str = "") -> str:
    """获取配置默认值：Streamlit Secrets → 环境变量 → secrets.toml文件 → 兜底"""
    # 1. Try Streamlit Secrets (for Streamlit Cloud)
    try:
        val = st.secrets.get(secret_key)
        if val:
            return val
    except Exception:
        pass

    # 2. Try environment variables
    val = os.environ.get(env_key)
    if val:
        return val

    # 3. Try reading from .streamlit/secrets.toml file
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            tomllib = None

    if tomllib:
        secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            with open(secrets_path, "rb") as f:
                data = tomllib.load(f)
                val = data.get(secret_key, "")
                if val:
                    return val

    # 4. Fallback
    return fallback

# Add parent to path for utils imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import memory as mem
from utils import asr as asr_utils
from utils import llm_utils
from utils import report
from utils import knowledge_base as kb

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
if "kb_loaded" not in st.session_state:
    st.session_state.kb_loaded = False
if "kb_stats" not in st.session_state:
    st.session_state.kb_stats = {"total": 0, "categories": {}}
if "kb_enabled" not in st.session_state:
    st.session_state.kb_enabled = True

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
        st.caption("🤖 **LLM（分析和编码）**")
        api_base = st.text_input(
            "LLM API 地址",
            value="https://api.deepseek.com/v1",
            help="用于文本分析和编码的大模型 API",
            placeholder="https://api.deepseek.com/v1",
            key="llm_base",
        )
        api_key = st.text_input(
            "API Key（共用）",
            type="password",
            value=_get_default("LLM_API_KEY", "LLM_API_KEY", ""),
            help="填一个有权限的 API Key",
            placeholder="sk-...",
            key="api_key",
        )
        llm_model = st.text_input(
            "LLM 模型",
            value="deepseek-chat",
            help="用于文本分析和编码的大模型",
            placeholder="deepseek-chat",
            key="llm_model",
        )

        st.divider()
        st.caption("🎤 **ASR 语音转写**")

        asr_provider = st.radio(
            "选择 ASR 服务商",
            options=["OpenAI Whisper", "阿里云录音文件识别"],
            horizontal=True,
            key="asr_provider",
        )

        if asr_provider == "OpenAI Whisper":
            st.caption("需要 OpenAI 兼容的 Whisper API Key")
            asr_base = st.text_input(
                "ASR API 地址",
                value="https://api.openai.com/v1",
                placeholder="https://api.openai.com/v1",
                key="asr_base",
            )
            asr_key = st.text_input(
                "ASR API Key",
                type="password",
                placeholder="sk-...",
                key="asr_key",
            )
            whisper_model = st.text_input(
                "Whisper 模型",
                value="whisper-1",
                placeholder="whisper-1",
                key="whisper_model",
            )
        else:
            st.caption("需要阿里云 AccessKey + OSS + DashScope API Key")
            ali_ak = st.text_input("AccessKey ID", value=_get_default("ALIYUN_AK", "ALIYUN_AK", ""), key="ali_ak")
            ali_sk = st.text_input("AccessKey Secret", type="password", value=_get_default("ALIYUN_SK", "ALIYUN_SK", ""), key="ali_sk")
            ali_oss_endpoint = st.text_input("OSS Endpoint", value=_get_default("ALIYUN_OSS_ENDPOINT", "ALIYUN_OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com"), key="ali_oss_endpoint")
            ali_oss_bucket = st.text_input("OSS Bucket 名称", value=_get_default("ALIYUN_OSS_BUCKET", "ALIYUN_OSS_BUCKET", "zwbssss"), key="ali_oss_bucket")
            ali_dashscope_key = st.text_input(
                "DashScope API Key",
                type="password",
                value=_get_default("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY", "sk-"),
                key="ali_dashscope_key",
                help="从 https://bailian.console.aliyun.com/ → API-KEY管理 获取"
            )

    st.markdown("---")
    st.caption("💡 推荐配置")
    st.caption("🧠 LLM: DeepSeek (`deepseek-chat`)")
    st.caption("🎤 OpenAI Whisper: 最简单，填 Key 即用")
    st.caption("🎤 阿里云ASR: 需 OSS + DashScope API Key（百炼平台获取）")

    st.markdown("---")
    st.caption("📁 输出：每次处理将自动生成")
    st.caption("1️⃣ **原话版.xlsx** — 问题与回答一一对应")
    st.caption("2️⃣ **初步编码版.xlsx** — 含主题编码/关键词/情感")

    # Knowledge Base status indicator
    st.markdown("---")
    kb_stats = st.session_state.kb_stats
    if kb_stats["total"] > 0:
        st.success(f"📚 知识库已加载 ({kb_stats['total']} 条目)")
        for cat, count in kb_stats["categories"].items():
            st.caption(f"  · {cat}: {count} 条")
    else:
        st.info("📚 知识库为空，前往「知识库管理」添加")


# ---------- Main UI ----------
st.title("🎙️ 访谈智能整理与分析系统")
st.markdown("""
上传访谈录音文件和结构化访谈框架，自动完成：
- **语音转写** → 自动识别受访者回答
- **语义匹配** → 将回答匹配到对应问题
- **双版本报告** → 同时输出原话版和初步编码版
""")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["📤 上传与处理", "📋 访谈问题编辑", "📚 知识库管理", "❓ 使用说明"])

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
            st.warning("请在左侧配置 LLM API Key", icon="⚠️")
        elif st.session_state.processing:
            st.warning("正在处理中，请稍候...", icon="⏳")

    # ASR status hint
    asr_provider_display = st.session_state.get("asr_provider", "OpenAI Whisper")
    if asr_provider_display == "OpenAI Whisper":
        asr_base_val = st.session_state.get("asr_base", "https://api.openai.com/v1")
        asr_key_val = st.session_state.get("asr_key", "")
        st.caption(f"🎤 ASR: {asr_provider_display} ({asr_base_val})")
    else:
        st.caption(f"🎤 ASR: {asr_provider_display}（需 OSS + NLS）")

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
        
        # === 实时状态面板 ===
        status_container = st.container(border=True)
        with status_container:
            col_prog, col_time = st.columns([3, 1])
            with col_prog:
                progress_bar = st.progress(0, text="⏳ 初始化...")
            with col_time:
                elapsed_placeholder = st.empty()
                elapsed_placeholder.caption("⏱️ 0秒")
            
            log_container = st.container(height=200)
            log_area = log_container.empty()
            log_lines = []
            
            def add_log(emoji, msg):
                from datetime import datetime
                ts = datetime.now().strftime("%H:%M:%S")
                log_lines.append(f"{ts}  {emoji} {msg}")
                log_area.markdown("  \n".join(log_lines[-15:]))

        import time as _time
        _start_t = _time.time()
        
        def update_elapsed():
            elapsed_placeholder.caption(f"⏱️ {int(_time.time() - _start_t)}秒")
        
        try:
            # Save uploaded file
            suffix = Path(audio_file.name).suffix
            tmp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(tmp_dir, f"audio{suffix}")
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

            add_log("📁", f"音频已保存: {audio_file.name} ({audio_file.size/1024:.0f}KB)")
            progress_bar.progress(5, text="音频已保存")
            update_elapsed()

            # ---------- Step 1: Audio Preprocessing ----------
            add_log("🔧", "步骤1/5: 音频预处理...")
            progress_bar.progress(10, text="音频预处理...")

            # Convert if not standard WAV
            wav_path = None
            if suffix.lower() != ".wav":
                if asr_utils.check_ffmpeg():
                    wav_path = asr_utils.convert_to_standard_wav(audio_path)
                    if wav_path:
                        add_log("✅", "音频已转为标准WAV格式")
                    else:
                        add_log("⚠️", "ffmpeg 转码失败，尝试直接使用原始文件")
                        wav_path = audio_path
                else:
                    add_log("⚠️", "未检测到 ffmpeg，使用原始文件")
                    wav_path = audio_path
            else:
                wav_path = audio_path

            file_size_mb = asr_utils.get_file_size_mb(wav_path)
            add_log("📊", f"音频大小: {file_size_mb:.1f}MB")

            progress_bar.progress(20, text="准备调用语音转写")
            update_elapsed()

            # ---------- Step 2: ASR ----------
            asr_provider = st.session_state.get("asr_provider", "OpenAI Whisper")
            asr_label = "阿里云ASR" if asr_provider == "阿里云录音文件识别" else "Whisper API"
            add_log("🎤", f"步骤2/5: 语音转写中（{asr_label}）...")
            progress_bar.progress(30, text="语音转写中（约1-5分钟）...")
            update_elapsed()

            if asr_provider == "阿里云录音文件识别":
                deps_ok, deps_msg = asr_utils.check_aliyun_asr_deps()
                if not deps_ok:
                    st.error(f"❌ {deps_msg}")
                    st.stop()

                ali_ak = st.session_state.get("ali_ak", "")
                ali_sk = st.session_state.get("ali_sk", "")
                ali_oss_endpoint = st.session_state.get("ali_oss_endpoint", "")
                ali_oss_bucket = st.session_state.get("ali_oss_bucket", "")
                ali_app_key = st.session_state.get("ali_app_key", "")

                if not all([ali_ak, ali_sk, ali_oss_endpoint, ali_oss_bucket, ali_app_key]):
                    st.error("❌ 请填写完整的阿里云 ASR 配置信息")
                    st.stop()

                # Create progress updater for Alibaba ASR
                def asr_progress(msg):
                    add_log("⏳", f"阿里云ASR: {msg}")
                    progress_bar.progress(35, text=msg)
                    update_elapsed()

                asr_result = asr_utils.transcribe_with_aliyun(
                    audio_path=wav_path,
                    access_key_id=ali_ak,
                    access_key_secret=ali_sk,
                    oss_endpoint=ali_oss_endpoint,
                    oss_bucket=ali_oss_bucket,
                    dashscope_api_key=ali_app_key,
                    progress_callback=asr_progress,
                )
            else:
                asr_base = st.session_state.get("asr_base", "https://api.openai.com/v1")
                asr_key = st.session_state.get("asr_key", "")
                whisper_model = st.session_state.get("whisper_model", "whisper-1")

                add_log("⏳", "调用 Whisper API 中（首次需下载模型，约30秒-2分钟）")
                update_elapsed()

                asr_result = asr_utils.transcribe_with_whisper_api(
                    audio_path=wav_path,
                    api_key=asr_key if asr_key else api_key,
                    base_url=asr_base,
                    model=whisper_model,
                    language="zh",
                )

            transcript = asr_result["full_text"]
            st.session_state.transcript = transcript

            if not transcript.strip():
                st.error("❌ 语音转写结果为空，请检查音频文件是否有效")
                st.stop()

            add_log("✅", f"转写完成：{len(transcript)} 字符")
            progress_bar.progress(50, text="转写完成")
            update_elapsed()

            # ---------- Step 3: LLM Analysis (with KB) ----------
            kb_status = "知识库已启用" if st.session_state.kb_stats["total"] > 0 else "知识库为空"
            add_log("🧠", f"步骤3/5: LLM分析与语义匹配（{kb_status}）...")
            progress_bar.progress(60, text="LLM 语义分析中（约30秒-1分钟）...")
            update_elapsed()

            client = llm_utils.get_client(api_key, base_url=api_base)

            # Extract answers and match to questions (KB-enhanced)
            matches = llm_utils.extract_answers_with_kb(
                client=client,
                transcript=transcript,
                questions=questions,
                model=llm_model,
            )

            add_log("✅", f"LLM 分析完成，共识别 {len(matches)} 个文本片段")
            update_elapsed()

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

            add_log("✅", f"匹配完成：{matched_count} 条回答匹配到 {answered_qs} 个问题")
            progress_bar.progress(80, text="匹配完成")
            update_elapsed()

            # ---------- Step 4: Generate Reports ----------
            add_log("📊", "步骤4/5: 生成双版本报告（原话版 + 编码版）...")
            progress_bar.progress(85, text="生成原话版报告...")
            update_elapsed()

            output_prefix = os.path.join(tmp_dir, f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            raw_path, coded_path = report.generate_dual_reports(
                questions=questions,
                interview_memory=interview_memory,
                output_prefix=output_prefix,
                client=client,
                model=llm_model,
            )

            add_log("✅", "原话版报告生成完成")
            progress_bar.progress(92, text="生成编码版报告（LLM编码中）...")
            update_elapsed()

            # Load files for download
            with open(raw_path, "rb") as f:
                raw_bytes = f.read()
            with open(coded_path, "rb") as f:
                coded_bytes = f.read()

            st.session_state.raw_report_bytes = raw_bytes
            st.session_state.coded_report_bytes = coded_bytes
            st.session_state.raw_report_path = raw_path
            st.session_state.coded_report_path = coded_path

            add_log("🎉", f"全部完成！耗时 {int(_time.time() - _start_t)} 秒")
            progress_bar.progress(100, text="✅ 全部完成！")
            elapsed_placeholder.caption(f"⏱️ {int(_time.time() - _start_t)}秒")

        except Exception as e:
            st.error(f"❌ 处理过程中发生错误: {e}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            add_log("❌", f"错误: {str(e)[:100]}")
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


# ===== Tab 3: Knowledge Base Management =====
with tab3:
    st.subheader("📚 知识库管理")
    st.markdown("""
    上传**往年优秀访谈记录**和**专业术语表**，让 LLM 在语义匹配和主题编码时参考这些信息，
    提高对领域特有表达的理解准确性和编码一致性。
    """)

    kb_check = st.session_state.kb_stats
    if kb_check["total"] > 0:
        st.success(f"✅ 当前知识库：{kb_check['total']} 条文档，{len(kb_check.get('categories', {}))} 个分类")

    # ── Section 1: Upload past interview records ──
    with st.expander("📄 上传优秀访谈记录", expanded=True):
        st.markdown("上传过往项目中优秀的访谈记录文本文件（.txt / .md），系统会自动分段索引。")
        interview_files = st.file_uploader(
            "选择访谈记录文件（可多选）",
            type=["txt", "md", "json"],
            accept_multiple_files=True,
            key="kb_interview_files",
        )
        if interview_files:
            total_chunks = 0
            for f in interview_files:
                content = f.read().decode("utf-8", errors="replace")
                kb_instance = kb.get_kb()
                chunks = kb_instance.add_text(
                    text=content,
                    category="访谈案例",
                    title=f.name,
                    source="历史访谈",
                )
                total_chunks += chunks
            if total_chunks > 0:
                st.session_state.kb_stats = kb_instance.count()
                st.session_state.kb_loaded = True
                st.success(f"✅ 已导入 {len(interview_files)} 个文件，共 {total_chunks} 个片段")
                st.rerun()

        # Manual input
        st.markdown("---")
        st.markdown("**或手动输入一段访谈案例：**")
        col_q, col_a = st.columns(2)
        with col_q:
            case_question = st.text_area("问题文本", key="kb_case_q", height=80,
                                         placeholder="例如：你平时玩什么类型的游戏？")
        with col_a:
            case_answer = st.text_area("受访者回答", key="kb_case_a", height=80,
                                       placeholder="例如：我主要玩王者荣耀和原神。")
        if st.button("➕ 添加此案例到知识库", key="add_case_btn"):
            if case_question and case_answer:
                kb_instance = kb.get_kb()
                kb_instance.add_interview_case(question=case_question, answer=case_answer)
                st.session_state.kb_stats = kb_instance.count()
                st.session_state.kb_loaded = True
                st.success("✅ 案例已添加！")
                st.rerun()
            else:
                st.warning("请同时填写问题和回答")

    # ── Section 2: Glossary management ──
    with st.expander("📖 专业术语与常用词汇表", expanded=True):
        st.markdown("添加你研究领域的专业术语，帮助 LLM 更准确理解受访者的表达。")

        # Add single term
        col1, col2, col3 = st.columns([3, 5, 1])
        with col1:
            term = st.text_input("术语", placeholder="例如：DAU", key="kb_term")
        with col2:
            definition = st.text_input("解释", placeholder="例如：日活跃用户数", key="kb_def")
        with col3:
            if st.button("➕ 添加", key="add_term_btn"):
                if term and definition:
                    kb_instance = kb.get_kb()
                    kb_instance.add_glossary_entry(term=term, definition=definition)
                    st.session_state.kb_stats = kb_instance.count()
                    st.session_state.kb_loaded = True
                    st.success(f"✅ 已添加：{term}")
                    st.rerun()
                else:
                    st.warning("请填写术语和解释")

        # Batch upload glossary file
        st.markdown("---")
        st.markdown("**批量导入术语表（每行一个：`术语：解释`）**")
        glossary_file = st.file_uploader(
            "上传术语表文件 (.txt / .md)",
            type=["txt", "md"],
            key="kb_glossary_file",
        )
        if glossary_file:
            content = glossary_file.read().decode("utf-8", errors="replace")
            kb_instance = kb.get_kb()
            count = 0
            for line in content.strip().split("\n"):
                line = line.strip()
                if "：" in line:
                    t, d = line.split("：", 1)
                    kb_instance.add_glossary_entry(t.strip(), d.strip())
                    count += 1
                elif ":" in line:
                    t, d = line.split(":", 1)
                    kb_instance.add_glossary_entry(t.strip(), d.strip())
                    count += 1
            if count > 0:
                st.session_state.kb_stats = kb_instance.count()
                st.session_state.kb_loaded = True
                st.success(f"✅ 已导入 {count} 个术语")
                st.rerun()
            else:
                st.warning("未识别到有效术语，请确保格式为「术语：解释」每行一个")

    # ── Section 3: View and manage ──
    with st.expander("📋 查看知识库内容"):
        kb_instance = kb.get_kb()
        stats = kb_instance.count()
        if stats["total"] == 0:
            st.info("知识库为空，请先添加上面内容")
        else:
            st.write(f"**总计：{stats['total']} 条文档**")
            for cat, cnt in stats.get("categories", {}).items():
                st.write(f"- {cat}: {cnt} 条")

            # Show all documents
            if st.button("展开所有内容"):
                for i, doc in enumerate(kb_instance.documents):
                    with st.container():
                        st.markdown(f"**{i+1}. [{doc['category']}] {doc['title']}**")
                        st.caption(doc["text"][:200] + ("..." if len(doc["text"]) > 200 else ""))
                    st.divider()

    # ── Section 4: Danger zone ──
    with st.expander("⚠️ 管理操作"):
        col_del1, col_del2 = st.columns(2)
        with col_del1:
            cat_to_del = st.selectbox(
                "删除特定分类",
                options=["全部"] + list(kb.get_kb().count().get("categories", {}).keys()),
            )
            if st.button("🗑️ 删除选中的分类", type="secondary"):
                if cat_to_del == "全部":
                    kb.reset_kb()
                    st.session_state.kb_stats = {"total": 0, "categories": {}}
                    st.success("✅ 知识库已清空")
                else:
                    removed = kb.get_kb().remove_by_category(cat_to_del)
                    st.session_state.kb_stats = kb.get_kb().count()
                    st.success(f"✅ 已删除 {removed} 条 {cat_to_del}")
                st.rerun()

        with col_del2:
            if st.button("🗑️ 清空整个知识库", type="primary"):
                kb.reset_kb()
                st.session_state.kb_stats = {"total": 0, "categories": {}}
                st.success("✅ 知识库已清空")
                st.rerun()


# ===== Tab 4: Usage Guide =====
with tab4:
    st.markdown("""
    ## 📖 使用说明

    ### 快速开始
    1. **左侧配置**：填写 API Base URL 和 API Key（支持 DeepSeek / OpenAI 等）
    2. **知识库**：前往「知识库管理」上传往年访谈记录与术语表（可选但推荐）
    3. **上传音频**：上传包含受访者回答的访谈录音（.m4a, .mp3, .wav）
    4. **编辑问题**：调整面试问题框架（支持增删改）
    5. **点击分析**：系统自动完成转写 → 匹配 → 编码 → 生成报告

    ### 双版本报告
    | 版本 | 内容 |
    |------|------|
    | **原话版** | 所有问题与受访者回答一一对应，未获取回答的问题黄色标注 |
    | **初步编码版** | 在原话基础上增加：主题编码、子编码、关键词、情感倾向 |

    ### 📚 RAG 知识库说明
    - **知识库作用**：在语义匹配和主题编码时，LLM 会参考你上传的过往访谈案例和专业术语
    - **推荐内容**：
      - 往年优秀访谈记录（存档关键问答对）
      - 专业术语表（如 DAU、ARPU、留存率、核心玩法等）
      - 常用词汇表（你研究领域特有的表达方式）
    - **不依赖外部服务**：纯关键词检索，无需额外 API 或数据库

    ### 支持的API
    - **DeepSeek (LLM)**: `https://api.deepseek.com/v1` + 模型 `deepseek-chat`
    - **OpenAI (ASR+LLM)**: `https://api.openai.com/v1` + 模型 `gpt-4o` / `whisper-1`
    - **Azure OpenAI**: 自定义 endpoint
    - **智谱开放平台**: `https://open.bigmodel.cn/api/paas/v4`

    ### ⚠️ 重要：ASR 和 LLM 可以分开配置
    - **LLM 用 DeepSeek**（分析和编码）+ **ASR 用 OpenAI Whisper**（语音转写）
    - 或者 **LLM 用 DeepSeek + ASR 用阿里云录音文件识别**
    - 在左侧「ASR 语音转写」处选择服务商并填写对应配置即可

    ### 🎤 ASR 服务商说明

    | 服务商 | 优点 | 需要配置 |
    |--------|------|---------|
    | **OpenAI Whisper** | 最简单，填 API Key 即用 | API Key + API 地址 |
    | **阿里云录音文件识别** | 中文识别更准，适合大量音频 | AccessKey + OSS + NLS AppKey |

    **阿里云配置获取：**
    1. AccessKey：在阿里云控制台 → RAM 用户 → 创建 AccessKey
    2. OSS：开通对象存储 OSS → 创建 Bucket → 获取 Endpoint
    3. NLS AppKey：开通智能语音交互服务 → 创建项目 → 获取 AppKey

    ### 注意事项
    - ⚠️ 单声道录音中主持人与受访者混合，系统会自动识别提取受访者回答
    - 🎯 保持问题文本清晰具体，有助于提高语义匹配准确率
    - 📂 报告为 .xlsx 格式，可用 Excel 或 WPS 打开
    """)

# ---------- Footer ----------
st.markdown("---")
st.caption("🎙️ 访谈智能整理与分析系统 v1.0 | 基于 LLM + Whisper 构建")