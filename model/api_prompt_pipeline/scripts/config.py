# scripts/config.py

import os

# ================= 椤圭洰璺緞閰嶇疆 =================
# 鍋囪鑴氭湰浣嶄簬 scripts/config.py锛屽悜涓婁袱绾ф壘鍒伴」鐩牴鐩綍
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)

# [杈撳叆婧?1] 瑙掕壊鎻愬彇缁撴灉 (Step 1 output)
# 鍖呭惈鏂囦欢濡? result_2013-05.csv.jsonl
ROLE_DIR = os.path.join(PROJECT_ROOT, "data", "role")

# [杈撳叆婧?2] 璐圭敤鍘熸枃鎻愬彇缁撴灉 (Step 2 output)
# 鍖呭惈鏂囦欢濡? result_2013-05.csv.jsonl
FEE_SENTENCE_DIR = os.path.join(PROJECT_ROOT, "data", "fee")

# [杈撳嚭鐩爣] 鏈€缁堢粨鏋勫寲缁撴灉
# 绋嬪簭浼氳嚜鍔ㄧ敓鎴愮被浼?result_final_2013-05.csv.jsonl 鐨勬枃浠?
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "final_extraction")

# 纭繚杈撳嚭鐩綍瀛樺湪
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= API 閰嶇疆 (SiliconFlow) =================

# 纭呭熀娴佸姩 API 鍦板潃 (OpenAI 鍏煎)
API_BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"

# API Keys 姹?
# 澶氳繘绋嬫椂浼氶殢鏈洪€夊彇 Key 杩涜璐熻浇鍧囪　
API_KEYS = [
    "sk-rcowefwvmcxxsdvkfbryhtnejpmmptwdylubyajtsatuzyda", 
    "sk-zfkaujxuobptiwkyalhuirduyosfrctorwvvqockouawbbau", 
    "sk-vicnnqzpstkrjgklmoczxbhxymxqxpykfyuuwosdqbdgtzdr", 
    "sk-ckpdksvhxkknpzhqzslnxgvxflhgammfrsdorahhldmzxtzw",
    "sk-mysaznauphxyngwnvpymnejkicyfyzhgtivvnwsdihzlrsvn", 
    "sk-ntntfavqmmhjwrcbhyxgunqmzavtpubokwsvyrearxstzmxl", 
    "sk-qngrqkmceqtdozqwrbcpihhrqcwccgkohprpdlmwwnvrxpsf", 
    "sk-futuqpisyvtdpcgcfjwrgjxgnlimudgyjipqvvibavyqzmja"
]

# 鐩爣妯″瀷
MODEL_NAME = "deepseek-ai/DeepSeek-V3" 

# ================= 杩愯鏃惰缃?=================

# 榛樿骞跺彂杩涚▼鏁?
# 寤鸿璁剧疆涓?API Key 鐨勬暟閲忔垨鑰?CPU 鏍稿績鏁?- 1
DEFAULT_WORKERS = 6

# 鎵瑰鐞嗘暟閲忛檺鍒?
# - 濡傛灉濉暣鏁?(濡?100): 鍙窇 100 鏉℃暟鎹敤浜庢祴璇曘€?
# - 濡傛灉濉?None: 璺戝畬 data/fee 鐩綍涓嬫墍鏈夊尮閰嶅埌鐨勬暟鎹?(鍏ㄩ噺妯″紡)銆?
DEFAULT_BATCH_LIMIT = None

# 429 闄愭祦 / 鑷姩閲嶈瘯绛栫暐
# 閬囧埌 "Too Many Requests" 鏃惰嚜鍔ㄧ瓑寰呭苟閲嶈瘯
RATE_LIMIT_INITIAL_DELAY = 2       # 棣栨绛夊緟 2 绉?
RATE_LIMIT_BACKOFF_FACTOR = 2      # 姣忔澶辫触绛夊緟鏃堕棿缈诲€?(2s -> 4s -> 8s...)
RATE_LIMIT_MAX_DELAY = 60          # 鏈€澶х瓑寰呮椂闂?60 绉?
RATE_LIMIT_MAX_RETRIES = 5         # 鏈€澶氶噸璇?5 娆★紝瓒呰繃鍒欐爣璁颁负澶辫触

# 閫氱敤瓒呮椂璁剧疆 (杩炴帴瓒呮椂, 璇诲彇瓒呮椂)
# DeepSeek-V3 澶勭悊澶嶆潅閫昏緫鍙兘闇€瑕佽緝闀挎椂闂达紝寤鸿璇诲彇瓒呮椂璁句负 120s
API_TIMEOUT = (10, 120)            

# ================= 楠岃瘉鎵撳嵃 =================
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Role Dir:     {ROLE_DIR}")
    print(f"Fee Dir:      {FEE_SENTENCE_DIR}")
    print(f"Output Dir:   {OUTPUT_DIR}")
    print(f"Run Limit:    {'ALL FILES' if DEFAULT_BATCH_LIMIT is None else DEFAULT_BATCH_LIMIT}")
