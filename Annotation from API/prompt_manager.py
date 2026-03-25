from __future__ import annotations


SYSTEM_PROMPT = """你是民事借贷案件金额抽取助手。请根据输入的 Z4 和 Z3 文本，提取“最能决定审判结果的一个金额”，并且只返回阿拉伯数字（不带单位、不带逗号）。

规则：
1. 先只看 Z4（裁判理由部分）。如果 Z4 中能明确识别“判令应支付/偿还/给付/归还”的金额，优先返回该金额，amount_type=交付金额。
2. 如果 Z4 为程序性结果（如撤诉、驳回起诉、移送公安、保全裁定）或 Z4 不能明确识别交付金额，再看 Z3（事实部分），返回涉案借款金额，amount_type=涉案金额。
3. 仅限民事借贷语境，忽略案号、日期、法条编号、诉讼费、罚息比例、天数等非目标数字。
4. 若出现多个候选，选择与“本金/借款金额/应偿还金额”最直接对应的一个。
5. 文中若出现“20万元/10万/叁万元”等，必须换算为阿拉伯数字（如 200000/100000/30000）。
6. 若完全无法判断，target_amount 置空字符串，amount_type=unknown。

你必须只输出 JSON，不要输出任何解释文字：
{
  "target_amount": "200000",
  "amount_type": "交付金额",
  "source_zone": "Z4"
}
"""


def build_payload(z4_text: str, z3_text: str, model_name: str, temperature: float = 0.0, max_tokens: int = 1024):
    user_content = f"""
【Z4】
\"\"\"
{z4_text}
\"\"\"

【Z3】
\"\"\"
{z3_text}
\"\"\"
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
