# 先安裝 sacrebleu
!pip install sacrebleu

import pandas as pd
import sacrebleu

# 讀取你的檔案
df = pd.read_csv("/content/kaxabu_translated_full_gpt5.csv")

# 設定欄位
predictions = df["kaxabu_mt"].astype(str).tolist()   # 模型翻譯
references = [df["噶哈巫語"].astype(str).tolist()]  # 參考翻譯

# 計算 Corpus BLEU
bleu = sacrebleu.corpus_bleu(predictions, references)
print(f"BLEU 分數：{bleu.score:.2f}")
