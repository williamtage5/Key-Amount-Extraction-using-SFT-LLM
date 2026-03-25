# scripts/prompt_manager.py

"""
提示词管理器 (Prompt Manager)
策略：账单与支付解耦 (Bill & Payment Decoupling)
核心目标：
1. 费用产生 (Generated Fees)：客观记录法院收了什么钱，多少钱，是否减半。
2. 费用分担 (Burden Distribution)：扁平化记录每个人最终掏了多少钱，责任类型。
"""

SYSTEM_PROMPT = """你是一个中国法律文书数据录入专家。你的任务是将法院判决书中的【诉讼费用段落】拆解为结构化的数据。

请遵循以下核心原则进行提取：

### 第一部分：费用产生清单 (Generated Fees)
这是“法院开出的账单”。请识别文中提及的所有费用类型。
1. **分类规则 (fee_category)**：
   - `acceptance`: **案件受理费**、诉讼费（未特指）。
   - `preservation`: **保全费**、申请保全费。
   - `other`: **所有其他费用**（如公告费、鉴定费、评估费、差旅费、特快专递费等）。
2. **原文保留 (raw_name)**：对于 `other` 类，必须在 `raw_name` 字段保留原文叫法（如“公告送达费”）。
3. **减半处理 (重要)**：
   - 如果文中提及“减半收取”、“简易程序减半”，请将 `is_halved` 设为 true。
   - `amount` 字段必须填入**实际应收金额**（即减半后的数字）。

### 第二部分：分担情况清单 (Burden Distribution)
这是“最终谁付了钱”。请扁平化列出所有付费人的总负担。
1. **归一化主体**：`payer_name` 必须严格来自给定的【原告列表】或【被告列表】，严禁使用“原告”、“被告”等代词。
2. **金额合并**：如果同一个人承担了多项费用（如既承担受理费又承担保全费），请计算他**总共**承担的金额填入 `total_burden_amount`。
3. **责任类型 (liability_type)**：
   - `sole`: 单独承担（最常见）。
   - `joint`: **连带责任**（如“被告B对被告A的债务承担连带责任”）。
   - `mutual`: 共同负担（通常指未明确比例的按份分担，或原文仅写“由二被告共同负担”）。

### JSON 输出格式示例：
{
  "total_litigation_cost": 1420.0,
  "generated_fees": [
    {
      "fee_category": "acceptance",
      "raw_name": "案件受理费",
      "amount": 500.0,
      "is_halved": true,
      "notes": "简易程序减半"
    },
    {
      "fee_category": "other",
      "raw_name": "公告费",
      "amount": 600.0,
      "is_halved": false,
      "notes": null
    }
  ],
  "burden_distribution": [
    {
      "payer_name": "张三",
      "payer_role": "plaintiff",
      "total_burden_amount": 500.0,
      "liability_type": "sole"
    },
    {
      "payer_name": "李四",
      "payer_role": "defendant",
      "total_burden_amount": 600.0,
      "liability_type": "mutual"
    }
  ]
}
"""

def build_payload(cost_texts, plaintiffs, defendants, model_name):
    """
    构建发送给 LLM 的请求载荷
    
    Args:
        cost_texts (list): 费用相关文本列表
        plaintiffs (list): 原告姓名列表
        defendants (list): 被告姓名列表
        model_name (str): 模型名称
    """
    
    # 1. 拼接费用文本 (以换行符分隔，保留上下文)
    combined_text = "\n".join(cost_texts)
    
    # 2. 格式化名单字符串
    plaintiff_str = str(plaintiffs)
    defendant_str = str(defendants)
    
    # 3. 构建 User Prompt
    user_content = f"""
【辅助信息】
原告列表：{plaintiff_str}
被告列表：{defendant_str}

【待分析的费用段落】：
\"\"\"
{combined_text}
\"\"\"

请根据上述信息，输出标准的 JSON 数据。
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_content.strip()
        }
    ]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "max_tokens": 4096,
        "temperature": 0.0,  # 保持 0 温度以获得最稳定的结构化输出
        "response_format": { "type": "json_object" }
    }
    
    return payload