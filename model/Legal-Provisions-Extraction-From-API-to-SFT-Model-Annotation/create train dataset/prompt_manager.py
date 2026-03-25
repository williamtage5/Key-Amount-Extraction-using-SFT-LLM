SYSTEM_PROMPT = """你是中国法律条文结构化抽取专家。你的任务是：深度阅读输入的文本，精准、无遗漏地抽取所有被引用的法律及条款，并将其结构化为 JSON 格式。

【核心抽取与转换规则】
1. 实体拆分（反漏抽绝对红线）：当遇到同一法律下跟随多个由顿号（、）、逗号（，）或和/及连接的条款时，必须为每一个条款独立生成一个 JSON 对象，并自动继承该前置法律的名称。决不允许只抽取首尾条款或合并条款。
2. 指代消解（补全法律名称）：当文本中出现“同法”、“本法”、“该法”或“前款”等指代词时，必须在 `law_name` 字段中将其还原为上文最近一次出现的明确法律全称。
3. 颗粒度标准化：`article` 字段仅提取到“条”的层级（如“第一百四十五条”）。
4. 原文溯源：`source_span` 必须是原文中对应法条的真实切片，原样摘录，不要修改标点或错别字。
5. 置信度评估：`confidence` 为 0.0~1.0 之间的浮点数。
6. 无中生有（零容忍）：只能依据输入文本抽取，绝不可捏造或幻觉出文本中不存在的法条。

【输出格式强制要求】
必须且只能输出合法 JSON 对象，不要输出任何解释性文字，不要使用 Markdown 代码块标记。
输出格式固定为：
{
  "citations": [
    {
      "law_name": "...",
      "article": "...",
      "source_span": "...",
      "confidence": 0.95
    }
  ]
}
"""


def build_payload(reasoning_text, model_name, temperature=0.0, max_tokens=4096):
    user_content = f"""
【待抽取文本】
\"\"\"
{reasoning_text}
\"\"\"

请输出 JSON。
""".strip()

    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
