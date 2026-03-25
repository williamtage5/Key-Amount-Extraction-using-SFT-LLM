cd "e:/Learning_journal_at_CUHK/CUHK_MSc_Project-32/Important_number_extraction/Annotation from API"


# 重新开始推理到5000条
python main_runner.py --input-dir "../data/yishen_partition" --limit 5000

# 接着续跑（需要将文件目录对齐）
python main_runner.py --input-dir "../data/yishen_partition" --run-name "<上次输出的时间戳文件夹名>" --limit 5000
