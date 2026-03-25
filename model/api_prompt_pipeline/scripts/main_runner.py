# scripts/main_runner.py

import os
import json
import multiprocessing
from tqdm import tqdm

# 导入配置和模块
from config import OUTPUT_DIR, DEFAULT_WORKERS, DEFAULT_BATCH_LIMIT
from data_merger import get_merged_data_generator
from extraction_service import process_merged_row

def get_completed_cases(output_dir):
    """
    [断点续传] 扫描输出目录，获取所有已成功处理的案号集合。
    """
    completed = set()
    if not os.path.exists(output_dir):
        return completed
    
    # 扫描 result_final_ 开头的文件
    files = [f for f in os.listdir(output_dir) if f.startswith("result_final_") and f.endswith(".jsonl")]
    
    print(f"[系统] 正在检查历史进度，扫描 {len(files)} 个输出文件...")
    
    for fname in files:
        fpath = os.path.join(output_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        record = json.loads(line)
                        # 只有 status 为 success 的才算真正完成
                        if record.get('status') == 'success':
                            meta = record.get('meta', {})
                            case_no = meta.get('case_no')
                            if case_no:
                                completed.add(case_no)
                    except:
                        continue
        except Exception as e:
            print(f"[警告] 读取历史文件 {fname} 出错: {e}")
            
    print(f"[系统] 已加载 {len(completed)} 个已完成的案号，将自动跳过。")
    return completed

def worker_task(args):
    """
    多进程 Worker 的包装函数。
    接收: (merged_data, original_filename)
    
    修改点：为了在最后打印输入的特征，我们将原始 input_data 也一并返回。
    返回: (processing_result, original_filename, original_input_data)
    """
    data, filename = args
    
    # 提取输入特征快照 (用于主进程打印 Demo)
    input_snapshot = data.get('input_data', {})
    
    # 调用核心业务逻辑
    result = process_merged_row(data)
    
    return result, filename, input_snapshot

def main():
    # 1. 准备环境
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 2. 加载断点信息
    completed_cases = get_completed_cases(OUTPUT_DIR)
    
    # 3. 准备任务队列
    print("[系统] 正在进行数据对齐与任务装载...")
    merger_gen = get_merged_data_generator()
    
    tasks = []
    skipped_count = 0
    
    # 遍历对齐后的数据流
    for merged_data, filename in merger_gen:
        case_no = merged_data['meta'].get('case_no')
        
        # [断点续传逻辑] 检查是否已做过
        if case_no in completed_cases:
            skipped_count += 1
            continue
            
        # 添加到任务列表
        tasks.append((merged_data, filename))
        
        # [批次限制逻辑]
        # 如果 DEFAULT_BATCH_LIMIT 为 None，则此条件不触发，处理所有数据
        if DEFAULT_BATCH_LIMIT is not None and len(tasks) >= DEFAULT_BATCH_LIMIT:
            break
            
    print(f"[系统] 任务准备完成：")
    print(f"  - 新增任务数: {len(tasks)}")
    print(f"  - 跳过历史任务: {skipped_count}")
    print(f"  - 启动进程数: {DEFAULT_WORKERS}")
    
    if not tasks:
        print("[系统] 没有新任务需要处理，程序结束。")
        return

    # 4. 启动多进程处理
    # 用于收集 5 个样本进行展示 (Result + Input Snapshot)
    demo_samples = []
    
    # 文件句柄池：{ 'result_final_2013-05.jsonl': open_file_handle }
    # 避免频繁打开关闭文件
    file_handles = {}

    pbar = tqdm(total=len(tasks), desc="结构化提取中", unit="条")
    
    try:
        with multiprocessing.Pool(processes=DEFAULT_WORKERS) as pool:
            # 使用 imap_unordered 提高效率
            for result, original_fname, input_snapshot in pool.imap_unordered(worker_task, tasks):
                
                # --- 结果写入逻辑 ---
                # 构造输出文件名: result_2013-05.csv.jsonl -> result_final_2013-05.csv.jsonl
                output_fname = original_fname.replace("result_", "result_final_")
                
                # 确保文件句柄已打开
                if output_fname not in file_handles:
                    out_path = os.path.join(OUTPUT_DIR, output_fname)
                    file_handles[output_fname] = open(out_path, 'a', encoding='utf-8')
                
                # 写入 (只写入 result，不写入 input_snapshot 以节省空间)
                file_handles[output_fname].write(json.dumps(result, ensure_ascii=False) + "\n")
                
                # --- 样本收集 ---
                # 收集成功样本用于展示，同时把 Input 特征挂载上去
                if result['status'] == 'success' and len(demo_samples) < 5:
                    sample_package = {
                        "result": result,
                        "inputs": input_snapshot
                    }
                    demo_samples.append(sample_package)
                
                # 更新进度条
                status_icon = "✅" if result['status'] == 'success' else "❌"
                latency = result.get('perf', {}).get('latency', 0)
                # 显示最近一条的耗时
                pbar.set_postfix_str(f"{status_icon} {latency}s")
                pbar.update(1)
                
    except KeyboardInterrupt:
        print("\n[用户中断] 正在停止...")
        pool.terminate()
    except Exception as e:
        print(f"\n[系统错误] {e}")
    finally:
        # 关闭所有文件句柄
        for fh in file_handles.values():
            fh.close()
        pbar.close()

    # 5. 打印对比样本 (Verification)
    # 这里将打印您要求的：输入的三个特征 + 输出的两个结果
    print("\n" + "="*80)
    print(f"【效果抽样展示】 (共展示 {len(demo_samples)} 条)")
    print("="*80)
    
    for i, item in enumerate(demo_samples, 1):
        res = item['result']
        inputs = item['inputs']
        meta = res.get('meta', {})
        extraction = res.get('extraction_result', {})
        
        print(f"\n[Sample {i}] 案号: {meta.get('case_no')}")
        
        # --- 打印输入特征 (Input Features) ---
        print(f"  [输入特征]:")
        print(f"    1. 原告列表: {inputs.get('plaintiffs')}")
        print(f"    2. 被告列表: {inputs.get('defendants')}")
        # 费用文本可能有换行，稍微处理一下显示
        cost_texts = inputs.get('cost_texts', [])
        print(f"    3. 费用原文: {cost_texts}")

        # --- 打印输出结果 (Output Results) ---
        print(f"  [提取结果]:")
        
        # 1. 账单层 (Bill)
        print(f"    >>> 产生的费用 (Generated Fees):")
        fees = extraction.get('generated_fees', [])
        if not fees: print("         (无数据)")
        for fee in fees:
            halved_mark = " (减半)" if fee.get('is_halved') else ""
            print(f"         - [{fee.get('fee_category')}] {fee.get('raw_name')}: {fee.get('amount')}元{halved_mark}")
            
        # 2. 支付层 (Burden)
        print(f"    >>> 最终分担 (Burden Distribution):")
        payers = extraction.get('burden_distribution', [])
        if not payers: print("         (无数据)")
        for p in payers:
            # 兼容可能的 key 缺失情况
            name = p.get('payer_name', '未知')
            role = p.get('payer_role', 'unknown')
            amt = p.get('total_burden_amount', 0)
            liab = p.get('liability_type', 'sole')
            print(f"         - {name} ({role}): {amt}元 [责任: {liab}]")
            
        print("-" * 60)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()