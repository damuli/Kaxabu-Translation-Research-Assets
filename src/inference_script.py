import pandas as pd
from openai import OpenAI
import time

# ✅ 讀取你的語料檔案
df = pd.read_csv("chinese_kaxabu.csv")

# ✅ 初始化 OpenAI 客戶端（請填入你的 API 金鑰）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ 使用你訓練完成的模型
model_name = "YOUR_FINETUNED_MODEL_ID_HERE"

# ✅ 新增一欄：存放模型翻譯
model_outputs = []

# ✅ 批次逐句翻譯
for i, row in df.iterrows():
    prompt = row["華語"]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        translation = response.choices[0].message.content.strip()
    except Exception as e:
        translation = f"ERROR: {e}"

    model_outputs.append(translation)

    # 為了安全與穩定，每次間隔 1 秒
    time.sleep(1)

# ✅ 把模型翻譯結果加到資料表中
df["模型翻譯"] = model_outputs

# ✅ 儲存成新檔案
df.to_csv("kaxabu_translation_with_model.csv", index=False)
