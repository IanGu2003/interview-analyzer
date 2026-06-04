# Prompt 版本变更记录

## v2 (当前) — 2025-06-03

**修改内容：**
- `extract_answers`: 重写为"服务于用户研究的中文访谈分析助手"定位，增加：
  - 更严格的"不准捏造"约束（original_text 必须逐字出现）
  - 识别主持人 vs 受访者的具体线索
  - 更精细的 score 评分标准（0.4/0.6/0.85 分档）
  - OFF_TOPIC 机制（不属于任何题目时不硬塞）
  - `{{KB_CONTEXT_SECTION}}` 占位符支持知识库注入
  - 详细示例（含误判防范说明）
- `code_response`: 重构为"质性研究编码助手"，增加：
  - 方法论原则（紧贴数据、数据驱动、一段可多码）
  - 双类型编码（descriptive + in_vivo）
  - confidence 三档评分标准（high/medium/low）
  - 字段约束（candidate_codes 1-3 条、sentiment n/a 兜底）
  - `{{Q_ID}}`、`{{QUESTION_TEXT}}`、`{{ANSWER_TEXT}}` 结构化占位符
  - `{{KB_CONTEXT_SECTION}}` 支持知识库注入
  - 保留 human_decision/human_code/note 供研究者复核
- `clean_text`: 重写为"极度克制的中文文本清理助手"，增加：
  - 可删除内容的精确枚举（不可扩展）
  - 绝对禁止事项清单
  - "那个/就是/然后"的歧义判断规则（指代 vs 填充）
  - `removals` 字段记录删除了什么
  - 3 个边界示例（含"那个"指代保留案例）
  - `{{TEXT}}` 占位符
- `llm_utils.py`: `render_prompt` 升级为通用模板引擎，支持任意 `{{VAR}}` 占位符
- `code_response_with_kb` 增加 `q_id` 参数
- 调用方改为 system prompt 内嵌所有变量，user message 简化

**改动原因：**
- 用户发现 v1 输出不够稳定：编码太粗糙、匹配太宽松、清理不够克制
- 需要更严格的约束来提升下游报告质量
- 需要支持研究者复核流程（human_* 字段）

**效果预期：**
- 语义匹配 precision 提升（OFF_TOPIC 减少误判）
- 编码质量提升（不再出现"未分类/待确认"兜底）
- 清理更克制（减少误删语义内容）

---

## v1 (2025-05-20)

**初始版本：**
- 基本语义匹配（extract_answers）
- 基础编码（theme_code + sub_code + keywords + sentiment）
- 简单文本清理（删除语气词）
- prompt 硬编码在 `utils/llm_utils.py` 中