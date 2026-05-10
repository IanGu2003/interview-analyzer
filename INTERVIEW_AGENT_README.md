# 访谈智能整理与分析Agent

## 项目概述

这是一个基于大语言模型的访谈数据智能处理系统，能够自动将音频转写、问题匹配、语气词清理，并生成结构化的访谈报告。

## 核心功能

1. **音频转写**：支持 mp3/wav/ogg/m4a 格式，自动获取时间戳
2. **文本清理**：自动剔除语气词（嗯、啊、那个等）
3. **智能匹配**：将回答匹配到对应的访谈问题
4. **相关性过滤**：识别并过滤跑题内容
5. **多格式输出**：支持 Excel/Word/Markdown/JSON

## 项目结构

```
src/
├── agents/
│   └── agent.py              # 主控Agent
└── tools/
    ├── speech_to_text.py     # 语音转写工具
    ├── text_cleaner.py       # 文本清理工具
    ├── segment_text.py       # 文本分段工具
    ├── relevance_classifier.py # 相关性判断工具
    ├── question_matcher.py   # 问题匹配工具
    ├── interview_memory.py    # 记忆管理工具
    └── postprocess_output.py  # 报告生成工具

config/
└── agent_llm_config.json      # 模型配置
```

## 使用方式

### 基本使用

```
用户：上传音频文件 + 访谈问题JSON
Agent：
  1. speech_to_text（转写音频）
  2. segment_with_timestamps（文本分段）
  3. relevance_classifier（相关性判断）
  4. text_cleaner（语气词清理）
  5. question_matcher（问题匹配）
  6. store_response（存储结果）
  7. postprocess_output（生成报告）
```

### 输入格式

**访谈问题JSON示例：**
```json
[
  {
    "question_id": "Q1",
    "question_text": "你平时使用哪些社交软件？",
    "possible_probes": ["为什么用这个？", "使用频率如何？"]
  }
]
```

### 输出格式

```
| 问题ID | 问题文本 | 受访者原话 | 时间戳 | 置信度 |
|--------|----------|-----------|--------|--------|
| Q1 | 你用什么社交软件？ | 我主要用微信 | 00:01:23 | 0.95 |
```

## 配置说明

- **模型配置**：config/agent_llm_config.json
- **默认模型**：doubao-seed-1-8-251228
- **温度参数**：0.7
- **最大Token**：10000

## 匹配策略

- **高置信度 (>= 0.7)**：直接匹配
- **模糊匹配 (0.4-0.7)**：需要人工确认
- **不匹配 (< 0.4)**：判定为跑题或闲聊

## 注意事项

1. 音频文件大小限制：≤100MB
2. 音频时长限制：≤2小时
3. 支持格式：WAV/MP3/OGG OPUS/M4A
