"""Come evaluate.py, ma genera con vLLM (molto più veloce). Stesse metriche e stessi file di output.

Va eseguito con l'ambiente vLLM separato:
    /root/vllm-env/bin/python training/evaluate_vllm.py --run qwen3-4b-r32
    /root/vllm-env/bin/python training/evaluate_vllm.py --base_only
"""
import argparse
import json
import time

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

from common import DATA_DIR, DEFAULT_MODEL, RUNS_DIR, prompt_text, read_jsonl
from evaluate import score_classify, score_extraction

p = argparse.ArgumentParser()
p.add_argument("--run")
p.add_argument("--base_only", action="store_true")
p.add_argument("--model", default=DEFAULT_MODEL)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--classify_n", type=int, default=500)
p.add_argument("--max_new", type=int, default=7000)
a = p.parse_args()
if not a.base_only and not a.run:
    raise SystemExit("specifica --run <nome> oppure --base_only")

out_dir = RUNS_DIR / (a.run or "base") / "eval"
out_dir.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.model)
llm = LLM(a.model, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.85,
          enable_lora=not a.base_only, max_lora_rank=32, seed=0)
lora = None if a.base_only else LoRARequest("ft", 1, str(RUNS_DIR / a.run / "adapter"))

metrics = {"run": a.run or "base", "model": a.model, "engine": "vllm"}
tasks = [("extraction", read_jsonl(DATA_DIR / "extraction_test.jsonl"), a.max_new),
         ("classify", read_jsonl(DATA_DIR / "classify_test.jsonl")[:a.classify_n], 128)]
for name, recs, max_new in tasks:
    if a.limit:
        recs = recs[:a.limit]
    print(f"[{name}] {len(recs)} esempi", flush=True)
    t0 = time.time()
    prompts = [prompt_text(tok, r["messages"]) for r in recs]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_new), lora_request=lora)
    preds = [o.outputs[0].text for o in outs]
    troncati = sum(o.outputs[0].finish_reason == "length" for o in outs)
    pairs = [(r["messages"][-1]["content"], pr) for r, pr in zip(recs, preds)]
    score = score_extraction(pairs) if name == "extraction" else score_classify(pairs)
    score["risposte_troncate"] = troncati
    score["minuti"] = round((time.time() - t0) / 60, 1)
    metrics[name] = score
    with open(out_dir / f"{name}_pred.jsonl", "w", encoding="utf-8") as f:
        for r, pr in zip(recs, preds):
            f.write(json.dumps({"meta": r["meta"], "gold": r["messages"][-1]["content"],
                                "pred": pr}, ensure_ascii=False) + "\n")
    print(json.dumps(score, indent=1), flush=True)

(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
print(f"\nRisultati in {out_dir}")
