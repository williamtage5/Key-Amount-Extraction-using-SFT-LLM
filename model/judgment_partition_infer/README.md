# judgment_partition_infer

Standalone inference bundle for partitioning a Chinese judgment document (or a truncated excerpt) into 7 zones (Z1..Z7) by predicting 6 boundaries.

## Install

```bash
pip install -r requirements.txt
```

## Input format (JSONL)

One JSON object per line. Required field: `text` (or `full_text`).

Example:
```json
{"sample_id":"demo_1","text":"...全文..."}
```

Optional fields `case_no` and `case_name` are passed through to outputs.

## Run inference (CLI)

From this folder:
```bash
python infer_cli.py --input examples/input.jsonl
```

Outputs are written under `output/<YYYYMMDD_HHMMSS>/` by default:
- `predictions.jsonl`
- `run_meta.json`

## Anchor behavior

Default: `--anchor auto`
- If anchors are detected, enforce:
  - boundary[0] = Z1 anchor ("号" within first 100 chars)
  - boundary[3] = Z4 anchor ("判决如下"/"如下判决")
- If anchors are missing/invalid, keep model boundaries and set `anchor_status` accordingly.

## Python API

```python
from judgment_partition_infer import Predictor

pred = Predictor()  # loads assets/best_model.pt + assets/vocab.json
out = pred.predict_text("...全文/片段...")
print(out["boundaries"])
```

# judgment_partition_infer

这是一个独立的推理工具包，用于通过预测 6 个边界位置，将中文裁判文书全文（或截断的片段）自动切分为 7 个固定结构分区（Z1~Z7）。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 输入格式 (JSONL)

输入文件必须为 JSONL 格式（每行一个独立的 JSON 对象）。
**必填字段**：`text`（系统也兼容读取 `full_text` 字段）。

**数据示例：**

```json
{"sample_id":"demo_1","text":"...全文..."}
```

> **注**：可选的元数据字段如 `case_no`（案号）和 `case_name`（案名）在处理过程中不会被修改，并会原样透传到输出结果中。

## 运行推理 (命令行方式)

请在当前工具包根目录下执行以下命令：

```bash
python infer_cli.py --input examples/input.jsonl

```

默认情况下，推理结果会保存在按时间戳自动生成的 `output/<YYYYMMDD_HHMMSS>/` 目录下，包含以下两个文件：

* `predictions.jsonl`：包含边界坐标、各个分区文本等最终预测结果。
* `run_meta.json`：本次推理任务的运行元数据及统计信息。

## 锚点规则 (Anchor Behavior)

系统默认启用自动锚点策略：`--anchor auto`

* **当检测到业务锚点时，强制执行以下约束：**
* `boundary[0]`（第 1 条边界）强制对齐至 **Z1 锚点**（即正文前 100 个字符内出现的最后一个“号”字）。
* `boundary[3]`（第 4 条边界）强制对齐至 **Z4 锚点**（匹配“判决如下”或“如下判决”）。


* **当锚点缺失或无效时：**
* 系统将直接保留模型预测的原始句子边界，并在输出结果中相应地更新 `anchor_status` 字段（标明锚点缺失）。



## Python API 调用 (代码内嵌方式)

如果你希望在自己的 Python 代码中直接调用该模型，可以使用以下接口：

```python
from judgment_partition_infer import Predictor

# 初始化预测器（会自动加载 assets/best_model.pt 和 assets/vocab.json）
pred = Predictor()  

# 传入文书全文或片段进行推理
out = pred.predict_text("...全文/片段...")

# 打印预测出的 6 个边界位置
print(out["boundaries"])
```
