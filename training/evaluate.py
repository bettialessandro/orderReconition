"""Valuta un adapter sui test set (brand esclusi dal training).

    python training/evaluate.py --run qwen3-4b-r32
    python training/evaluate.py --run qwen3-4b-r32 --limit 20      # prova veloce
    python training/evaluate.py --base_only                         # modello base, come riferimento

Scrive in runs/<run>/eval/: predizioni JSONL e metrics.json.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DATA_DIR, DEFAULT_MODEL, RUNS_DIR, prompt_text, read_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from controlli import abbina_righe, controlla_taglie  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="nome della cartella in runs/")
    p.add_argument("--base_only", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0, help="max esempi per task (0 = tutti)")
    p.add_argument("--classify_n", type=int, default=500)
    p.add_argument("--batch", type=int, default=8)
    return p.parse_args()


# --------------------------------------------------------------------------
def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        return None


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().upper()


def num(x):
    try:
        return round(float(x), 2)
    except Exception:
        return None


def row_key(r):
    return (norm(r.get("articolo")), norm(r.get("colore")))


def taglie_set(r):
    return sorted((norm(t.get("taglia")), num(t.get("qta"))) for t in (r.get("taglie") or []))


def score_extraction(pairs):
    """Righe abbinate con colore tollerante (controlli.abbina_righe); 'righe_f1_colore_identico'
    e' la misura rigida di prima. Il controllo taglie e' misurato come rilevatore di errori."""
    m = dict(n=0, json_ok=0, blocchi_esatti=0, neg_n=0, neg_ok=0, tp=0, tp_rig=0, fp=0, fn=0,
             campi_n=0, costo_ok=0, qta_ok=0, taglie_ok=0, taglie_ko_segnalate=0, taglie_ok_segnalate=0)
    for gold_txt, pred_txt in pairs:
        m["n"] += 1
        gold = json.loads(gold_txt)["righe"]
        pred = parse_json(pred_txt)
        if isinstance(pred, dict) and isinstance(pred.get("righe"), list):
            m["json_ok"] += 1
            pred = [r for r in pred["righe"] if isinstance(r, dict)]
        else:
            pred = []
        if not gold:
            m["neg_n"] += 1
            m["neg_ok"] += int(not pred)
        coppie, mancanti, in_piu = abbina_righe(gold, pred)
        m["tp"] += len(coppie); m["tp_rig"] += sum(1 for *_, e in coppie if e)
        m["fp"] += len(in_piu); m["fn"] += len(mancanti)
        exact = not mancanti and not in_piu
        for g, p, _ in coppie:
            m["campi_n"] += 1
            c = num(g.get("costo")) == num(p.get("costo"))
            q = num(g.get("qta")) == num(p.get("qta"))
            t = taglie_set(g) == taglie_set(p)
            m["costo_ok"] += c; m["qta_ok"] += q; m["taglie_ok"] += t
            if controlla_taglie(p):
                m["taglie_ok_segnalate" if t else "taglie_ko_segnalate"] += 1
            exact &= c and q and t
        m["blocchi_esatti"] += exact

    def f1(tp):
        prec, rec = tp / max(1, tp + m["fp"] + (m["tp"] - tp)), tp / max(1, tp + m["fn"] + (m["tp"] - tp))
        return prec, rec, 2 * prec * rec / max(1e-9, prec + rec)
    prec, rec, f = f1(m["tp"])
    taglie_ko = m["campi_n"] - m["taglie_ok"]
    return {
        "esempi": m["n"], "positivi": m["n"] - m["neg_n"], "negativi": m["neg_n"],
        "json_valido": m["json_ok"] / m["n"],
        "blocco_esatto": m["blocchi_esatti"] / m["n"],
        "blocco_esatto_positivi": (m["blocchi_esatti"] - m["neg_ok"]) / max(1, m["n"] - m["neg_n"]),
        "negativi_corretti": m["neg_ok"] / max(1, m["neg_n"]),
        "righe_precision": prec, "righe_recall": rec, "righe_f1": f,
        "righe_f1_colore_identico": f1(m["tp_rig"])[2],
        "costo_acc": m["costo_ok"] / max(1, m["campi_n"]),
        "qta_acc": m["qta_ok"] / max(1, m["campi_n"]),
        "taglie_acc": m["taglie_ok"] / max(1, m["campi_n"]),
        "controllo_taglie_errori_intercettati": m["taglie_ko_segnalate"] / max(1, taglie_ko),
        "controllo_taglie_falsi_allarmi": m["taglie_ok_segnalate"],
    }


def score_classify(pairs):
    fields = ["categoria", "subcategoria", "base_color"]
    ok = {f: 0 for f in fields}; all_ok = 0; js = 0
    for gold_txt, pred_txt in pairs:
        g = json.loads(gold_txt); p = parse_json(pred_txt)
        if not isinstance(p, dict):
            continue
        js += 1
        hit = [norm(g.get(f)) == norm(p.get(f)) for f in fields]
        for f, h in zip(fields, hit):
            ok[f] += h
        all_ok += all(hit)
    n = max(1, len(pairs))
    return {"esempi": len(pairs), "json_valido": js / n, "tutti_e_tre": all_ok / n,
            **{f"{f}_acc": ok[f] / n for f in fields}}


# --------------------------------------------------------------------------
@torch.no_grad()
def generate(model, tok, records, batch, max_new):
    prompts = [prompt_text(tok, r["messages"]) for r in records]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    outs = [None] * len(prompts)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            outs[i] = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  {min(s + batch, len(order))}/{len(order)}", flush=True)
    return outs


def main():
    a = parse_args()
    if not a.base_only and not a.run:
        raise SystemExit("specifica --run <nome> oppure --base_only")
    out_dir = RUNS_DIR / (a.run or "base") / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map="cuda")
    if not a.base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, RUNS_DIR / a.run / "adapter").merge_and_unload()
    model.eval()

    metrics = {"run": a.run or "base", "model": a.model}
    tasks = [("extraction", read_jsonl(DATA_DIR / "extraction_test.jsonl"), 7000),
             ("classify", read_jsonl(DATA_DIR / "classify_test.jsonl")[:a.classify_n], 128)]
    for name, recs, max_new in tasks:
        if a.limit:
            recs = recs[:a.limit]
        print(f"[{name}] {len(recs)} esempi")
        t0 = time.time()
        preds = generate(model, tok, recs, a.batch if name == "extraction" else 64, max_new)
        pairs = [(r["messages"][-1]["content"], p) for r, p in zip(recs, preds)]
        score = score_extraction(pairs) if name == "extraction" else score_classify(pairs)
        score["minuti"] = round((time.time() - t0) / 60, 1)
        metrics[name] = score
        with open(out_dir / f"{name}_pred.jsonl", "w", encoding="utf-8") as f:
            for r, p in zip(recs, preds):
                f.write(json.dumps({"meta": r["meta"], "gold": r["messages"][-1]["content"],
                                    "pred": p}, ensure_ascii=False) + "\n")
        print(json.dumps(score, indent=1))

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"\nRisultati in {out_dir}")


if __name__ == "__main__":
    main()
