# scripts/convert_to_autodl_txt.py

import os
import json
import glob
from config import OUTPUT_DIR, ROLE_DIR, FEE_SENTENCE_DIR, PROJECT_ROOT

# ================= 配置区域 =================
TARGET_DIR = os.path.join(PROJECT_ROOT, "data", "training_ready")
TARGET_FILE = os.path.join(TARGET_DIR, "dataset_for_autodl.txt")

# 安全阈值：超过此长度的数据（Instruction+Input+Output）将被自动丢弃
# 显存杀手通常都是几万字的超长乱码，8000字对于法律费用提取任务通常足够了
MAX_TOTAL_LENGTH = 8000 

# === 升级版系统指令 ===
# 清晰描述任务目标 + 明确 JSON 结构定义
SYSTEM_INSTRUCTION = (
    "你是一个中国法律文书数据结构化专家。你的任务是根据给定的【原告名单】、【被告名单】及【费用段落】，"
    "提取案件受理费及其他诉讼费用的产生明细与最终分担情况。\n"
    "请严格输出标准 JSON 格式，不要包含 Markdown 标记或额外解释。输出结构需包含以下字段：\n"
    "1. total_litigation_cost (float): 案件受理费总金额。\n"
    "2. cost_details (list): 费用产生明细，包含 name(费用名称), amount(金额), paid_by(预交人)。\n"
    "3. cost_sharing (list): 费用分担结果，包含 bearer(承担人), amount(承担金额), type(连带/独立等)。"
)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_source_map(file_path, data_type):
    """加载源数据 (保持不变)"""
    data_map = {}
    if not os.path.exists(file_path):
        return data_map
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    record = json.loads(line)
                    if record.get('status') != 'success': continue
                    meta = record.get('meta', {})
                    case_no = meta.get('case_no')
                    if not case_no: continue
                    annotation = record.get('annotation', {})
                    
                    if data_type == 'role':
                        data_map[case_no] = {
                            'plaintiffs': annotation.get('plaintiff', []),
                            'defendants': annotation.get('defendant', [])
                        }
                    elif data_type == 'fee':
                        data_map[case_no] = annotation.get('legal_cost_sentences', [])
                except: continue
    except: pass
    return data_map

def format_input_text_flat(plaintiffs, defendants, cost_texts):
    """
    构造单行输入，清洗掉物理换行符
    """
    p_str = str(plaintiffs).replace('\n', '').replace('\r', '')
    d_str = str(defendants).replace('\n', '').replace('\r', '')
    
    if cost_texts:
        # 清洗掉每段文本内部的物理换行
        clean_costs = [str(t).replace('\n', '').replace('\r', '').strip() for t in cost_texts]
        # 使用显式字符 \n 连接，方便模型理解这是多段话，但物理上不换行
        c_str = "\\n".join(clean_costs)
    else:
        c_str = ""
    
    return f"【原告名单】：{p_str}\\n【被告名单】：{d_str}\\n【费用段落】：{c_str}"

def main():
    ensure_dir(TARGET_DIR)
    
    final_files = glob.glob(os.path.join(OUTPUT_DIR, "result_final_*.jsonl"))
    print(f"[系统] 扫描到 {len(final_files)} 个标注文件，准备处理...")
    
    valid_count = 0
    skipped_count = 0
    skipped_details = []

    with open(TARGET_FILE, 'w', encoding='utf-8') as outfile:
        for final_path in final_files:
            final_filename = os.path.basename(final_path)
            original_filename = final_filename.replace("result_final_", "result_")
            
            role_path = os.path.join(ROLE_DIR, original_filename)
            fee_path = os.path.join(FEE_SENTENCE_DIR, original_filename)
            
            role_map = load_source_map(role_path, 'role')
            fee_map = load_source_map(fee_path, 'fee')
            
            if not role_map or not fee_map: continue

            try:
                with open(final_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            record = json.loads(line)
                            # 基本校验
                            if record.get('status') != 'success': continue
                            
                            case_no = record.get('meta', {}).get('case_no')
                            role_data = role_map.get(case_no)
                            cost_texts = fee_map.get(case_no)
                            
                            if not role_data or not cost_texts: continue
                            
                            extraction_result = record.get('extraction_result')
                            # 如果结果为空，或者结果不是字典，跳过
                            if not extraction_result or not isinstance(extraction_result, dict):
                                continue
                            
                            # === 核心处理 ===
                            
                            # 1. 构造 Prompt
                            instruction_line = f"指令：{SYSTEM_INSTRUCTION}"
                            flat_input = format_input_text_flat(role_data['plaintiffs'], role_data['defendants'], cost_texts)
                            question_line = f"问：{flat_input}"
                            
                            # 2. 构造 Output (确保是单行 JSON)
                            output_content = json.dumps(extraction_result, ensure_ascii=False)
                            answer_line = f"答：{output_content}"
                            
                            # === 3. 关键步骤：长度安检 (排雷) ===
                            # 计算这一条数据的总字符数
                            total_chars = len(instruction_line) + len(question_line) + len(answer_line)
                            
                            if total_chars > MAX_TOTAL_LENGTH:
                                skipped_count += 1
                                if skipped_count <= 5: # 只打印前5个超长的例子
                                    print(f"⚠️ [警告] 跳过超长数据: 案号 {case_no}, 长度 {total_chars} > {MAX_TOTAL_LENGTH}")
                                continue
                            
                            # 写入文件 (确保每部分占一行，且最后空一行)
                            outfile.write(instruction_line + "\n")
                            outfile.write(question_line + "\n")
                            outfile.write(answer_line + "\n")
                            outfile.write("\n")
                            
                            valid_count += 1
                            
                        except Exception as e:
                            # 某些数据解析错误直接跳过
                            continue
            except: continue

    print("-" * 30)
    print(f"[处理完成] 目标文件: {TARGET_FILE}")
    print(f"✅ 有效录入: {valid_count} 条")
    print(f"🚫 过滤异常: {skipped_count} 条 (长度超过 {MAX_TOTAL_LENGTH} 或格式错误)")
    print("-" * 30)
    print("【下一步操作】:")
    print("1. 请将 dataset_for_autodl.txt 上传至 AutoDL 的 /root/LLaMA-Factory/data/ 目录")
    print("2. 运行 sh DD.sh 进行格式转换")
    print("3. 使用 Python check_full_data.py 再次验证，确保万无一失")

if __name__ == "__main__":
    main()