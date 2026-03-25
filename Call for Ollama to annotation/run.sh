# 不随机：按顺序跑 X 条
python "Call for Ollama to annotation\run_ollama_amount.py" --input-dir "data\yishen_partition" --output-base-dir "data\ollama annotation" --limit 10 --workers 2 --progress-every 1


# 随机：随机抽样跑 X 条（可复现）
python "Call for Ollama to annotation\run_ollama_amount.py" `
  --input-dir "data\yishen_partition" `
  --output-base-dir "data\ollama annotation" `
  --random-sample `
  --seed 42 `
  --limit 5 `
  --workers 2 `
  --progress-every 1

