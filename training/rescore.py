"""Ricalcola metrics.json dalle predizioni gia' salvate (senza rigenerare con il modello).

    python training/rescore.py --run qwen3-4b-r32
    python training/rescore.py --run base
"""
import argparse
import json

from common import RUNS_DIR
from evaluate import score_classify, score_extraction

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
a = p.parse_args()
d = RUNS_DIR / a.run / "eval"
path = d / "metrics.json"
metrics = json.loads(path.read_text()) if path.exists() else {"run": a.run}
for name, fn in [("extraction", score_extraction), ("classify", score_classify)]:
    preds = [json.loads(l) for l in open(d / f"{name}_pred.jsonl", encoding="utf-8")]
    score = fn([(r["gold"], r["pred"]) for r in preds])
    old = metrics.get(name, {})
    for k in ("minuti", "risposte_troncate"):
        if k in old:
            score[k] = old[k]
    metrics[name] = score
path.write_text(json.dumps(metrics, indent=1))
print(json.dumps(metrics, indent=1))
