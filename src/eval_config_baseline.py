# === Kaxabu MT 評測：Exact Match / Edit Distance / BLEU（Colab 版）===
import pandas as pd, math
from collections import Counter

# ===== 路徑與欄位設定 =====
CSV_PATH = "/content/kaxabu_translated_full_gpt5.csv"  # 你的檔案路徑
REF_COL  = "噶哈巫語"      # 參考答案（gold/reference）
HYP_COL  = "kaxabu_mt"     # 模型輸出（hypothesis/system）

# （可選）如果要同檔輸出到雲端硬碟，請先手動在前面掛載 drive 再改這裡
OUT_SENT_PATH = "/content/kaxabu_eval_results.csv"
OUT_SUMM_PATH = "/content/kaxabu_eval_summary.csv"

# ===== 讀檔 =====
df = pd.read_csv(CSV_PATH)
assert REF_COL in df.columns and HYP_COL in df.columns, "找不到指定欄位，請檢查 REF_COL / HYP_COL"

work = df[[REF_COL, HYP_COL]].copy().fillna("")
work.columns = ["ref_kaxabu", "hyp_kaxabu"]

# ====== 基本工具 ======
def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m+1):
            tmp = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = tmp
    return dp[m]

def tok(s: str):
    # 預設以空白分詞；若你有既定斷詞規則，可改成字元級：list(s)
    return s.strip().split()

def ngrams(tokens, n):
    L = len(tokens)
    return [tuple(tokens[i:i+n]) for i in range(L-n+1)] if L >= n else []

def sentence_bleu(candidate, references, max_n=4, eps=1e-9):
    weights = [1.0/max_n]*max_n
    # modified n-gram precisions
    p_ns = []
    for n in range(1, max_n+1):
        cand_ngr = Counter(ngrams(candidate, n))
        max_ref = Counter()
        for ref in references:
            ref_counts = Counter(ngrams(ref, n))
            for ng in ref_counts:
                if ref_counts[ng] > max_ref[ng]:
                    max_ref[ng] = ref_counts[ng]
        overlap = sum(min(cand_ngr[ng], max_ref.get(ng, 0)) for ng in cand_ngr)
        total = max(sum(cand_ngr.values()), 1)
        p_ns.append(overlap / total)
    # brevity penalty
    c = len(candidate)
    ref_lens = [len(r) for r in references]
    r = min(ref_lens, key=lambda rl: (abs(rl - c), rl)) if ref_lens else 0
    bp = 1.0 if c > r else (math.exp(1 - r / c) if c > 0 else 0.0)
    # geometric mean
    log_prec = sum(w * math.log(p + eps) for w, p in zip(weights, p_ns))
    return bp * math.exp(log_prec)

def corpus_bleu(cands, refs_list, max_n=4):
    eps = 1e-9
    weights = [1.0/max_n]*max_n
    overlap_counts = [0]*max_n
    cand_counts = [0]*max_n
    cand_len = 0
    ref_len = 0
    for cand, refs in zip(cands, refs_list):
        cand_len += len(cand)
        rlens = [len(r) for r in refs]
        r = min(rlens, key=lambda x: (abs(x - len(cand)), x)) if rlens else 0
        ref_len += r
        for n in range(1, max_n+1):
            cand_ngr = Counter(ngrams(cand, n))
            max_ref = Counter()
            for ref in refs:
                rc = Counter(ngrams(ref, n))
                for ng in rc:
                    if rc[ng] > max_ref[ng]:
                        max_ref[ng] = rc[ng]
            overlap_counts[n-1] += sum(min(cand_ngr[ng], max_ref.get(ng, 0)) for ng in cand_ngr)
            cand_counts[n-1] += sum(cand_ngr.values())
    p_ns = [(overlap_counts[i]/cand_counts[i]) if cand_counts[i] > 0 else 0.0 for i in range(max_n)]
    log_prec = sum(w * math.log(p + eps) for w, p in zip(weights, p_ns))
    bp = 1.0 if cand_len > ref_len else (math.exp(1 - ref_len / max(cand_len, 1)) if cand_len > 0 else 0.0)
    return bp * math.exp(log_prec)

# ====== 1) Exact Match ======
work["exact_match"] = (work["ref_kaxabu"].str.strip() == work["hyp_kaxabu"].str.strip())
exact_rate = work["exact_match"].mean()

# ====== 2) Edit Distance（字元級；含標準化）======
work["edit_distance"] = [levenshtein(r, h) for r, h in zip(work["ref_kaxabu"], work["hyp_kaxabu"])]
# 以「參考長度」為分母；若參考為空，退回用hyp長度，避免除以0
work["edit_distance_norm_ref"] = [
    d / (len(r) if len(r) > 0 else max(len(h), 1))
    for d, r, h in zip(work["edit_distance"], work["ref_kaxabu"], work["hyp_kaxabu"])
]
# （可選）對稱分母：兩者平均長度
work["edit_distance_norm_avg"] = [
    d / max(((len(r) + len(h)) / 2), 1)
    for d, r, h in zip(work["edit_distance"], work["ref_kaxabu"], work["hyp_kaxabu"])
]

# ====== 3) BLEU ======
work["sentence_bleu"] = [
    sentence_bleu(tok(h), [tok(r)], max_n=4)
    for r, h in zip(work["ref_kaxabu"], work["hyp_kaxabu"])
]
corpus_bleu_score = corpus_bleu(
    [tok(h) for h in work["hyp_kaxabu"].tolist()],
    [[tok(r)] for r in work["ref_kaxabu"].tolist()],
    max_n=4
)

# ====== 輸出與摘要 ======
summary = pd.DataFrame({
    "Metric": [
        "Exact Match Rate",
        "Avg Edit Distance",
        "Avg Normalized Edit Distance (ref)",
        "Avg Normalized Edit Distance (avg)",
        "Average Sentence BLEU",
        "Corpus BLEU"
    ],
    "Value": [
        work["exact_match"].mean(),
        work["edit_distance"].mean(),
        work["edit_distance_norm_ref"].mean(),
        work["edit_distance_norm_avg"].mean(),
        work["sentence_bleu"].mean(),
        corpus_bleu_score
    ]
})

print(summary.to_string(index=False))
work.to_csv(OUT_SENT_PATH, index=False)
summary.to_csv(OUT_SUMM_PATH, index=False)
print(f"\n已輸出逐句結果：{OUT_SENT_PATH}")
print(f"已輸出總結指標：{OUT_SUMM_PATH}")
