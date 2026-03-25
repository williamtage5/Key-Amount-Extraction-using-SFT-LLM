# Important Number Extraction Project

## English Version

### 1. Project Overview
This repository is an end-to-end pipeline for Chinese judgment processing, with a focus on structured extraction tasks such as loan amount extraction.

Current practical online inference path (already implemented and tested):
- Use a quantized Q4 GGUF model (`Qwen2.5-7B-Instruct-merged-Q4_K_M.gguf`).
- Import it into Ollama as `Law-Qwen-4bit`.
- Run batch extraction with:
  - `Call for Ollama to annotation/run_ollama_amount.py`

The script now writes results in this structure:
- `data/ollama annotation/<run_timestamp>/<year>/<year-month>.json`

Each output record contains:
- `source_file`
- `case_no`
- `target_amount`
- `amount_type`
- `source_zone`

---

### 2. Top-Level Subprojects and Their Roles
- `data/`
  - Project datasets and outputs.
  - Key folders include:
  - `yishen/`: raw judgment CSV files.
  - `yishen_partition/`: partitioned JSON data with zones (Z1..Z7).
  - `sft_amount_dataset/`: SFT-style training dataset for amount task.
  - `ollama annotation/`: Ollama inference outputs.
- `model/`
  - Model assets and model-centric pipelines.
  - `judgment_partition_infer/`: standalone partition inference package (predict judgment boundaries and zones).
  - `api_prompt_pipeline/`: API-based structured extraction pipeline for fee/role-style tasks.
  - `Legal-Provisions-Extraction-From-API-to-SFT-Model-Annotation/`: full pipeline workspace for API extraction, SFT dataset generation, and SFT/Ollama-based annotation scripts.
  - `SFT model/`: local model artifacts:
  - `raw_model/`: merged raw HF model.
  - `gguf model/`: GGUF model and Modelfile.
- `Segment/`
  - Runner to partition `data/yishen/*.csv` into zoned JSON under `data/yishen_partition/`.
- `Annotation from API/`
  - API-based amount extraction implementation and prompt design.
- `Call for Ollama to annotation/`
  - Local Ollama-based amount extraction implementation (current recommended inference entrypoint).
- `Quantilize the raw model/`
  - Quantization scripts (llama.cpp flow) and Ollama model creation flow for local deployment.

---

### 3. Current Ollama Model Status
- Quantized GGUF file:
  - `model/SFT model/gguf model/Qwen2.5-7B-Instruct-merged-Q4_K_M.gguf`
  - Size: `4,683,073,568` bytes (~4.36 GiB)
- Hugging Face hosted model repo:
  - `https://huggingface.co/WilliamCHN/Key_Amount_Extractor_Qwen`
- Hugging Face file links:
  - `https://huggingface.co/WilliamCHN/Key_Amount_Extractor_Qwen/resolve/main/Qwen2.5-7B-Instruct-merged-Q4_K_M.gguf`
  - `https://huggingface.co/WilliamCHN/Key_Amount_Extractor_Qwen/resolve/main/Modelfile.from-gguf`
- Ollama model name:
  - `Law-Qwen-4bit`
- Typical Ollama executable path in this project setup:
  - `E:\ollama\ollama.exe`

---

### 4. How to Use the Current Quantized Model with Ollama

#### Step A: Verify Ollama and model
Use PowerShell:

```powershell
& "E:\ollama\ollama.exe" list
```

You should see `Law-Qwen-4bit` in the model list.

#### Step B: (Optional) Download GGUF from Hugging Face

```powershell
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='WilliamCHN/Key_Amount_Extractor_Qwen', filename='Qwen2.5-7B-Instruct-merged-Q4_K_M.gguf', local_dir='model/SFT model/gguf model')"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='WilliamCHN/Key_Amount_Extractor_Qwen', filename='Modelfile.from-gguf', local_dir='model/SFT model/gguf model')"
```

#### Step C: (Optional) Re-import GGUF into Ollama
If model is missing, create from GGUF:

```powershell
& "E:\ollama\ollama.exe" create Law-Qwen-4bit -f "E:\Learning_journal_at_CUHK\CUHK_MSc_Project-32\Important_number_extraction\model\SFT model\gguf model\Modelfile.from-gguf"
```

#### Step D: Run batch inference (sequential first X samples)

```powershell
python "Call for Ollama to annotation\run_ollama_amount.py" --input-dir "data\yishen_partition" --output-base-dir "data\ollama annotation" --limit 10 --workers 2 --progress-every 1
```

#### Step E: Run batch inference (random X samples)

```powershell
python "Call for Ollama to annotation\run_ollama_amount.py" --input-dir "data\yishen_partition" --output-base-dir "data\ollama annotation" --random-sample --seed 42 --limit 10 --workers 2 --progress-every 1
```

#### Step F: Output location and format
After run, check:
- `data/ollama annotation/<timestamp>/<year>/<year-month>.json`

Output example:

```json
[
  {
    "source_file": "data\\yishen_partition\\2013\\2013-05.json",
    "case_no": "（2013）温平商初字第74号",
    "target_amount": "200000",
    "amount_type": "涉案金额",
    "source_zone": "Z3"
  }
]
```

#### Step G: Final console summary
The script now prints only final stats in `[Done]`:
- `input_dir`
- `output_run_dir`
- `model`
- `workers`
- `limit`
- `total_submitted`
- `success`
- `failed`
- `skipped`
- `timestamp`

---

### 5. Quantization Script (llama.cpp path)
Main script:
- `Quantilize the raw model/quantize_qwen_gguf_q4km_with_llamacpp.ps1`

What it does:
- Converts HF model to F16 GGUF.
- Quantizes to `Q4_K_M`.
- Verifies GGUF header.
- Creates Ollama model via Modelfile.

---

### 6. Recommended Practical Flow
1. Partition source judgments if needed (`Segment/` pipeline).
2. Run Ollama amount extraction with `run_ollama_amount.py`.
3. Inspect outputs under `data/ollama annotation/<timestamp>/...`.
4. Use extracted results for evaluation / downstream data assembly.

---

## 中文版本

### 1. 项目概述
这是一个中文裁判文书结构化处理项目，当前重点任务是“金额抽取”。

目前已经跑通并建议使用的本地推理路径：
- 使用量化后的 Q4 GGUF 模型（`Qwen2.5-7B-Instruct-merged-Q4_K_M.gguf`）。
- 通过 Ollama 导入并调用模型名 `Law-Qwen-4bit`。
- 使用脚本：
  - `Call for Ollama to annotation/run_ollama_amount.py`

当前输出目录结构：
- `data/ollama annotation/<运行时间戳>/<年份>/<年月>.json`

每条输出只保留 5 个字段：
- `source_file`
- `case_no`
- `target_amount`
- `amount_type`
- `source_zone`

---

### 2. 根目录各子项目作用
- `data/`
  - 数据输入与输出总目录。
  - 重点子目录：
  - `yishen/`：原始裁判文书 CSV。
  - `yishen_partition/`：分区后的 JSON（含 Z1..Z7）。
  - `sft_amount_dataset/`：金额任务 SFT 数据集。
  - `ollama annotation/`：Ollama 推理结果。
- `model/`
  - 模型与模型流水线相关目录。
  - `judgment_partition_infer/`：文书分区推理模型（预测分区边界）。
  - `api_prompt_pipeline/`：基于外部 API 的结构化抽取流水线。
  - `Legal-Provisions-Extraction-From-API-to-SFT-Model-Annotation/`：从 API 抽取到 SFT 数据构建、再到 SFT/Ollama 标注的完整工作区。
  - `SFT model/`：本地模型资产：
  - `raw_model/`：合并后的原始模型。
  - `gguf model/`：GGUF 与 Modelfile。
- `Segment/`
  - 把 `data/yishen` 的 CSV 分区为 `data/yishen_partition` 的执行入口。
- `Annotation from API/`
  - API 版本的金额抽取实现与提示词工程。
- `Call for Ollama to annotation/`
  - Ollama 本地模型调用版金额抽取（当前主入口）。
- `Quantilize the raw model/`
  - 量化与模型导入脚本（llama.cpp 路线）。

---

### 3. 当前 Ollama 模型状态
- GGUF 文件：
  - `model/SFT model/gguf model/Qwen2.5-7B-Instruct-merged-Q4_K_M.gguf`
  - 大小：`4,683,073,568` 字节（约 4.36 GiB）
- Hugging Face 模型仓库：
  - `https://huggingface.co/WilliamCHN/Key_Amount_Extractor_Qwen`
- Hugging Face 文件直链：
  - `https://huggingface.co/WilliamCHN/Key_Amount_Extractor_Qwen/resolve/main/Qwen2.5-7B-Instruct-merged-Q4_K_M.gguf`
  - `https://huggingface.co/WilliamCHN/Key_Amount_Extractor_Qwen/resolve/main/Modelfile.from-gguf`
- Ollama 模型名：
  - `Law-Qwen-4bit`
- 常用 Ollama 路径（本项目环境）：
  - `E:\ollama\ollama.exe`

---

### 4. 如何使用 Ollama 调用当前量化模型

#### A. 检查模型是否已导入

```powershell
& "E:\ollama\ollama.exe" list
```

若列表中有 `Law-Qwen-4bit`，即可直接推理。

#### B. （可选）从 Hugging Face 下载 GGUF

```powershell
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='WilliamCHN/Key_Amount_Extractor_Qwen', filename='Qwen2.5-7B-Instruct-merged-Q4_K_M.gguf', local_dir='model/SFT model/gguf model')"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='WilliamCHN/Key_Amount_Extractor_Qwen', filename='Modelfile.from-gguf', local_dir='model/SFT model/gguf model')"
```

#### C. （可选）从 GGUF 重新导入
如果模型不存在：

```powershell
& "E:\ollama\ollama.exe" create Law-Qwen-4bit -f "E:\Learning_journal_at_CUHK\CUHK_MSc_Project-32\Important_number_extraction\model\SFT model\gguf model\Modelfile.from-gguf"
```

#### D. 顺序抽取前 X 条

```powershell
python "Call for Ollama to annotation\run_ollama_amount.py" --input-dir "data\yishen_partition" --output-base-dir "data\ollama annotation" --limit 10 --workers 2 --progress-every 1
```

#### E. 随机抽取 X 条

```powershell
python "Call for Ollama to annotation\run_ollama_amount.py" --input-dir "data\yishen_partition" --output-base-dir "data\ollama annotation" --random-sample --seed 42 --limit 10 --workers 2 --progress-every 1
```

#### F. 输出格式与位置
运行完成后查看：
- `data/ollama annotation/<时间戳>/<年份>/<年月>.json`

示例：

```json
[
  {
    "source_file": "data\\yishen_partition\\2013\\2013-05.json",
    "case_no": "（2013）温平商初字第74号",
    "target_amount": "200000",
    "amount_type": "涉案金额",
    "source_zone": "Z3"
  }
]
```

#### G. 终端最终返回
脚本最后 `[Done]` 只返回统计信息，不再附带样本详情：
- `input_dir`
- `output_run_dir`
- `model`
- `workers`
- `limit`
- `total_submitted`
- `success`
- `failed`
- `skipped`
- `timestamp`

---

### 5. 量化脚本说明（llama.cpp 路线）
脚本：
- `Quantilize the raw model/quantize_qwen_gguf_q4km_with_llamacpp.ps1`

功能：
- HF 模型转 F16 GGUF。
- GGUF 量化为 `Q4_K_M`。
- 校验 GGUF 头。
- 自动生成 Modelfile 并执行 `ollama create`。

---

### 6. 建议执行顺序
1. 需要时先做文书分区（`Segment/`）。
2. 直接用 Ollama 脚本做金额抽取（`run_ollama_amount.py`）。
3. 在 `data/ollama annotation/<时间戳>/...` 检查结果。
4. 将结果用于评估、数据闭环或下游训练构造。
#   K e y - A m o u n t - E x t r a c t i o n - u s i n g - S F T - L L M  
 