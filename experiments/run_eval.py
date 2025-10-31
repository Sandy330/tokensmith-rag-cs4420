# experiments/run_eval.py
import os, time, json, logging, yaml
import pandas as pd

# Import pipeline entrypoint
from src.main import get_answer

# ------------ minimal logger ------------
logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    logger.addHandler(h)

# ------------ load config.yaml ------------
with open("config/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# ------------ safe args object ------------
class SafeArgs:
    """Return None for unspecified attributes so get_answer() doesn't crash."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __getattr__(self, _):
        return None

# Fill common fields the pipeline may look for
args = SafeArgs(
    mode="chat",
    index_prefix="book_index",      # <— must match the one you used at index time
    pdf_dir="data/chapters",
    top_k=cfg.get("top_k", 5),
    system_prompt_mode=cfg.get("system_prompt_mode", "baseline"),
    golden_pages=None,
    use_golden=False,
)


# ------------ questions ------------
df = pd.read_csv("question_sets/core.csv")
results = []
print(f"Loaded {len(df)} questions...")

for i, row in df.iterrows():
    q = row["question"]
    t0 = time.time()
    # IMPORTANT: this signature is (question, cfg, args, logger)
    ans = get_answer(q, cfg, args, logger)
    t1 = time.time()
    latency = round((t1 - t0) * 1000.0, 1)

    print(f"[{i+1}/{len(df)}] ⏱ {latency} ms  Q: {q}")
    print(f"   A: {repr(ans)[:160]}")

    results.append({
        "question": q,
        "latency_ms": latency,
        "answer": ans
    })

os.makedirs("experiments/results", exist_ok=True)
out_path = "experiments/results/run1.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

avg = sum(r["latency_ms"] for r in results) / max(1, len(results))
print(f"\n✅ Saved {len(results)} results to {out_path}")
print(f"📊 Average latency: {avg:.1f} ms")
