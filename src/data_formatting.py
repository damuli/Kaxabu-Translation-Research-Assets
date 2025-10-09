import pandas as pd
import json

# === 步驟 1: 讀取 CSV 檔案並初始化 DataFrame ===
# 請注意：在 GitHub 上公開時，程式碼不能依賴 Colab 的環境，
# 因此我們假設檔案在同一個目錄中。
try:
    # 讀取您上傳的 CSV 檔案
    df = pd.read_csv('chinese_kaxabu.csv')
except FileNotFoundError:
    print("錯誤：找不到 chinese_kaxabu.csv。請確認檔案路徑。")
    # 如果找不到檔案，則停止後續操作
    exit() 

# === 步驟 2: （可選）數據檢查，確保欄位存在 ===
# 這一部分取代了您在 Colab 中的 df.head() 和 df.columns
assert '華語' in df.columns and '噶哈巫語' in df.columns, "CSV 檔案中缺少 '華語' 或 '噶哈巫語' 欄位，請檢查。"
print(f"成功讀取 {len(df)} 筆資料，開始進行格式轉換。")


# === 步驟 3: 數據轉換為微調用的 JSON Lines 格式 ===
output = []

for _, row in df.iterrows():
    prompt = row['華語']
    completion = row['噶哈巫語']
    output.append({
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
    })


# === 步驟 4: 儲存成 jsonl 檔案 ===
OUTPUT_FILENAME = "kaxabu_finetune_format.jsonl" # 建議使用更專業的名稱
with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
    for item in output:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"數據轉換完成，檔案已儲存為：{OUTPUT_FILENAME}")
