# Fee Identification and Labeled

This project builds a data pipeline to extract structured litigation-fee information from Chinese court judgments using an LLM. It merges pre-extracted role information (plaintiff/defendant lists) with fee-related text, sends the merged data to an LLM for structured extraction, and then converts the results into a training-ready dataset for fine-tuning.

## What It Does

- **Merge inputs**: joins role data and fee-text data by `case_no`.
- **LLM extraction**: produces structured JSON for fee generation and burden distribution.
- **Resume & batch**: supports checkpointing and multi-process execution.
- **Training export**: converts final JSONL results into an instruction dataset for fine-tuning.

## Directory Structure

- `data/role/` – Role extraction outputs (JSONL)
- `data/fee/` – Fee sentence extraction outputs (JSONL)
- `data/final_extraction/` – Final structured extraction outputs (JSONL)
- `data/training_ready/` – Training dataset output (TXT)
- `data_sample/` – One-sample copies of each data file (for format reference)
- `scripts/` – Pipeline code

## Sample Data (Format Reference)

To help readers understand the data formats, this repo includes a **one-record sample** for each data file in `data_sample/` (same filenames as the originals).

- `data_sample/role/*.jsonl`
- `data_sample/fee/*.jsonl`
- `data_sample/final_extraction/*.jsonl`
- `data_sample/training_ready/dataset_for_autodl.txt`

## Data Formats

### Role / Fee JSONL
Each line is one JSON object with at least:

- `meta.case_no`: case identifier
- `annotation`: extracted content
- `status`: `success` or `failed`

Role files typically include:
- `annotation.plaintiff`: list of plaintiffs
- `annotation.defendant`: list of defendants

Fee files typically include:
- `annotation.legal_cost_sentences`: list of fee-related text segments

### Final Extraction JSONL
Each line is one JSON object with:

- `meta.case_no`
- `status`
- `extraction_result`: structured fees and burden distribution

### Training Dataset TXT
The training file is a repeating 4-line pattern:

1. `instruction: ...`
2. `question: ...`
3. `answer: ...`
4. blank line

## How to Run

> Make sure you have Python 3.10+.

Install dependencies:

```
pip install requests tqdm
```

Run the main extraction pipeline:

```
python scripts/main_runner.py
```

Convert final results into a training dataset:

```
python scripts/convert_to_autodl_txt.py
```

## Configuration Notes

- Edit `scripts/config.py` to set API keys and model parameters.
- **Do not commit real API keys** to a public repository.

## Notes

- The pipeline expects filenames to align across `data/role` and `data/fee`.
- If you only want to process a subset, set `DEFAULT_BATCH_LIMIT` in `scripts/config.py`.


# 费用识别与标注 (Fee Identification and Labeled)

本项目构建了一个自动化数据流水线，旨在利用大语言模型 (LLM) (外部API) 从中国法院判决书中提取结构化的诉讼费用信息。该流程通过合并预先提取的角色信息（原告/被告名单）与费用相关文本，调用 LLM 进行结构化提取，并最终将提取结果转换为可直接用于模型微调的训练数据集。

该任务的下游任务是根据AI标注的费用分配结果，微调一个小的大模型，从而能够构造出一个独立的、功能单一的费用的结构化提取器。

## 项目核心功能

* **输入合并**：根据案号 (`case_no`) 自动匹配并合并角色数据与费用文本数据。
* **LLM 结构化提取**：生成关于费用产生项及负担分配细节的结构化 JSON 数据。
* **断点续传与批处理**：支持检查点 (Checkpoint) 机制及多进程并行执行，确保大规模处理的稳定性。
* **训练数据导出**：将最终的 JSONL 提取结果转换为适用于指令微调的 TXT 数据集格式。

## 目录结构

* `data/role/` – 角色提取输出目录 (JSONL)
* `data/fee/` – 费用语句提取输出目录 (JSONL)
* `data/final_extraction/` – 最终结构化提取结果目录 (JSONL)
* `data/training_ready/` – 转换后的训练数据集目录 (TXT)
* `data_sample/` – 各数据文件的单条样本副本（供格式参考）
* `scripts/` – 流水线核心逻辑代码

## 示例数据 (格式参考)

为了方便理解数据结构，本仓库在 `data_sample/` 目录下提供了每个中间环节文件的**单条记录样本**（文件名与原始数据保持一致）：

* `data_sample/role/*.jsonl`
* `data_sample/fee/*.jsonl`
* `data_sample/final_extraction/*.jsonl`
* `data_sample/training_ready/dataset_for_autodl.txt`

## 数据格式说明

### 角色 / 费用 JSONL

每行代表一个 JSON 对象，至少包含以下字段：

* `meta.case_no`: 案号，作为唯一标识符。
* `annotation`: 提取的具体内容。
* `status`: 提取状态（`success` 或 `failed`）。

**角色文件**通常包含：

* `annotation.plaintiff`: 原告名单。
* `annotation.defendant`: 被告名单。

**费用文件**通常包含：

* `annotation.legal_cost_sentences`: 与费用相关的原始文本片段列表。

### 最终提取结果 JSONL

每行包含：

* `meta.case_no`
* `status`
* `extraction_result`: 结构化的费用详情及负担分配比例。

### 训练数据集 TXT

训练文件采用重复的 **4 行模式** 组织：

1. `instruction: ...` (指令)
2. `question: ...` (问题)
3. `answer: ...` (答案/期望输出)
4. （空行）

## 运行指南

> **环境要求**：Python 3.10 或更高版本。

1. **安装必要依赖**：
```bash
pip install requests tqdm

```


2. **运行主提取流水线**：
```bash
python scripts/main_runner.py

```


3. **将提取结果转换为训练数据集**：
```bash
python scripts/convert_to_autodl_txt.py

```



## 配置说明

* 请编辑 `scripts/config.py` 文件以配置 API 密钥（API Keys）及模型相关参数。
* **安全提示**：请勿将包含真实 API 密钥的代码提交至公共远程仓库。

## 注意事项

* 流水线依赖于 `data/role` 和 `data/fee` 目录下的文件名对齐，请确保输入文件名一致。
* 如需仅测试少量样本，请修改 `scripts/config.py` 中的 `DEFAULT_BATCH_LIMIT` 参数。

---

希望这份文档能帮助你和你的团队更高效地使用该工具。如果你需要我为你补充 `scripts/config.py` 的具体配置项说明，或者需要我帮你编写 `requirements.txt` 文件，随时告诉我！