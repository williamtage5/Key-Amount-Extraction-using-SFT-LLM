# scripts/data_merger.py

import os
import json
from config import ROLE_DIR, FEE_SENTENCE_DIR

def load_role_map(role_file_path):
    """
    [构建阶段]
    将整个 Role JSONL 文件加载到内存作为哈希表。
    Key: case_no
    Value: annotation dict (包含原告/被告信息)
    """
    role_map = {}
    if not os.path.exists(role_file_path):
        return role_map

    try:
        with open(role_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # 我们只关心 Step 1 提取成功的记录
                    if record.get('status') != 'success':
                        continue
                    
                    meta = record.get('meta', {})
                    case_no = meta.get('case_no')
                    
                    if case_no:
                        role_map[case_no] = record.get('annotation', {})
                        
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[错误] 加载角色文件失败 {role_file_path}: {e}")
    
    return role_map

def get_merged_data_generator():
    """
    [探测阶段]
    生成器：yield (合并后的数据字典, 来源文件名)
    """
    
    # 1. 扫描 Fee 目录下的所有文件
    if not os.path.exists(FEE_SENTENCE_DIR):
        print(f"[错误] 费用目录不存在: {FEE_SENTENCE_DIR}")
        return

    # 找到所有以 result_ 开头的 jsonl 文件
    fee_files = [f for f in os.listdir(FEE_SENTENCE_DIR) if f.endswith('.jsonl') and f.startswith('result_')]
    
    print(f"[系统] 发现 {len(fee_files)} 个费用文件待处理。")

    for filename in fee_files:
        fee_path = os.path.join(FEE_SENTENCE_DIR, filename)
        role_path = os.path.join(ROLE_DIR, filename) # 假设文件名是一一对应的

        # 检查对应的 Role 文件是否存在
        if not os.path.exists(role_path):
            print(f"[跳过] 未找到对应的角色文件: {filename}")
            continue

        # 步骤 A: 加载 Role 数据到内存 (Hash Map)
        # print(f"[系统] 正在加载角色表: {filename}...")
        role_lookup = load_role_map(role_path)
        
        if not role_lookup:
            print(f"[跳过] 该文件未提取到有效的角色数据: {filename}")
            continue

        # 步骤 B: 流式读取 Fee 数据并进行合并 (Join)
        # print(f"[系统] 开始数据对齐: {filename}...")
        
        try:
            with open(fee_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        fee_record = json.loads(line)
                        
                        # 过滤：只处理 Step 2 费用提取成功的记录
                        if fee_record.get('status') != 'success':
                            continue
                        
                        meta = fee_record.get('meta', {})
                        case_no = meta.get('case_no')
                        
                        # --- HASH JOIN 核心逻辑 ---
                        # 通过 'case_no' 将 Fee 记录与 Role 记录匹配
                        if case_no and case_no in role_lookup:
                            role_annotation = role_lookup[case_no]
                            fee_annotation = fee_record.get('annotation', {})
                            
                            # 获取 Step 2 提取出的具体费用段落
                            # 注意：根据你的数据示例，Key 是 "legal_cost_sentences"
                            cost_texts = fee_annotation.get('legal_cost_sentences', [])
                            
                            # 如果没有费用文本，就没有分析的必要
                            if not cost_texts:
                                continue

                            # 构造最终的合并数据包
                            merged_payload = {
                                "meta": meta, # 保留原始元数据 (source_file, text_length, truncated 等)
                                "input_data": {
                                    # 来自 Role 文件
                                    "plaintiffs": role_annotation.get('plaintiff', []),
                                    "defendants": role_annotation.get('defendant', []),
                                    # 来自 Fee 文件
                                    "cost_texts": cost_texts
                                }
                            }
                            
                            # Yield 数据 + 文件名 (供主程序决定写入哪个文件)
                            yield merged_payload, filename
                            
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[错误] 读取费用文件失败 {fee_path}: {e}")

if __name__ == "__main__":
    # --- 调试代码块 ---
    import json
    print("--- 正在测试数据合并逻辑 (Data Merger) ---")
    
    # 获取生成器
    gen = get_merged_data_generator()
    
    # 尝试打印前 3 条合并后的数据
    for i, (data, fname) in enumerate(gen):
        print(f"\n[{i+1}] 来源文件: {fname}")
        print(f"    案号: {data['meta']['case_no']}")
        
        # 打印合并后的具体内容 (使用 ensure_ascii=False 正常显示中文)
        inputs = data['input_data']
        print(f"    [原告]: {json.dumps(inputs['plaintiffs'], ensure_ascii=False)}")
        print(f"    [被告]: {json.dumps(inputs['defendants'], ensure_ascii=False)}")
        print(f"    [费用原文]: {json.dumps(inputs['cost_texts'], ensure_ascii=False)}")
        
        if i >= 2: 
            print("\n测试结束，仅展示前 3 条。")
            break