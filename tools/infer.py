"""
Pipeline di produzione: PDF (o Excel) di un brand -> righe d'ordine JSON, con verifica del totale.

  python tools/infer.py conferma.pdf [-o out.json]
  python tools/infer.py --eval dataset/documenti_test.jsonl      # valutazione a documento intero
  python tools/infer.py --eval ... --mock                         # prova del flusso senza modello

Il modello e' chiamato tramite un endpoint compatibile OpenAI (vLLM, Ollama, llama.cpp server)
in locale. Configurazione con variabili d'ambiente:
  SLM_URL   (default http://localhost:8000/v1)
  SLM_MODEL (default orderreconition)

Il testo viene estratto e diviso in blocchi con le STESSE funzioni usate per costruire il dataset
(build_dataset.pdf_pages / excel_pages / chunks): non cambiarle senza rigenerare il dataset.
"""
import os, sys, json, re, argparse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_dataset as B

SLM_URL = os.environ.get("SLM_URL", "http://localhost:8000/v1")
SLM_MODEL = os.environ.get("SLM_MODEL", "orderreconition")


def call_model(text, mock=False):
    if mock:
        return {"righe": []}
    body = {"model": SLM_MODEL, "temperature": 0, "max_tokens": 8192,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": B.SYSTEM_EXTRACT},
                         {"role": "user", "content": text}]}
    req = urllib.request.Request(SLM_URL.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        content = json.loads(r.read())["choices"][0]["message"]["content"]
    try:
        out = json.loads(content)
        return out if isinstance(out.get("righe"), list) else {"righe": [], "errore": "formato"}
    except json.JSONDecodeError:
        return {"righe": [], "errore": "json_non_valido", "raw": content[:500]}


def merge_rows(per_chunk):
    """Unisce le righe dei blocchi. Uno stesso articolo+colore puo' comparire in due blocchi
    (pagina spezzata): le taglie vengono unite tenendo la quantita' massima per taglia."""
    merged = {}
    for rows in per_chunk:
        for r in rows:
            k = (B.norm(r.get("articolo", "")), B.norm(r.get("colore", "")))
            if k not in merged:
                merged[k] = {**r, "taglie": list(r.get("taglie") or [])}
                continue
            m = merged[k]
            sizes = {t["taglia"]: t["qta"] for t in m["taglie"]}
            for t in r.get("taglie") or []:
                sizes[t["taglia"]] = max(sizes.get(t["taglia"], 0), t["qta"])
            m["taglie"] = [{"taglia": s, "qta": q} for s, q in sizes.items()]
    for m in merged.values():
        m["qta"] = sum(t["qta"] for t in m["taglie"])
    return list(merged.values())


def process(path, mock=False):
    pages = B.pdf_pages(path) if path.lower().endswith(".pdf") else B.excel_pages(path)
    full = "\n".join(pages)
    if len(full.strip()) < 50:
        return {"file": path, "errore": "nessun testo estraibile: PDF scansionato, serve OCR", "righe": []}
    blocks = B.chunks(pages)
    outs = [call_model(t, mock) for t in blocks]
    righe = merge_rows([o["righe"] for o in outs])
    nums_full = B.parse_numbers(full)
    tot = round(sum((r.get("costo") or 0) * (r.get("qta") or 0) for r in righe), 2)
    controlli = {
        "blocchi": len(blocks),
        "errori_blocchi": [i for i, o in enumerate(outs) if o.get("errore")],
        "totale_righe": tot,
        "totale_trovato_nel_documento": bool(righe) and B.has_num(nums_full, tot),
        "righe_con_costo_non_trovato": [r["articolo"] for r in righe
                                        if not B.has_num(nums_full, r.get("costo") or -1)],
    }
    controlli["da_verificare"] = (not controlli["totale_trovato_nel_documento"]
                                  or bool(controlli["errori_blocchi"])
                                  or bool(controlli["righe_con_costo_non_trovato"]))
    return {"file": path, "righe": righe, "controlli": controlli}


def evaluate(jsonl, mock=False):
    key = lambda r: (B.norm(r["articolo"]), B.norm(r["colore"] or ""))
    sizes = lambda r: sorted((str(t["taglia"]), int(t["qta"])) for t in r["taglie"])
    tp = fp = fn = exact = doc_ok = verified = n = 0
    for line in open(jsonl, encoding="utf-8"):
        d = json.loads(line); n += 1
        res = process(os.path.join(B.SRC, d["meta"]["sorgente"]), mock)
        gold = {key(r): r for r in d["righe_attese"]}
        pred = {key(r): r for r in res["righe"]}
        hit = gold.keys() & pred.keys()
        tp += len(hit); fp += len(pred.keys() - gold.keys()); fn += len(gold.keys() - pred.keys())
        ex = sum(1 for k in hit if abs((pred[k].get("costo") or 0) - gold[k]["costo"]) < 0.01
                 and sizes(pred[k]) == sizes(gold[k]))
        exact += ex
        doc_ok += int(ex == len(gold) == len(pred))
        verified += int(res.get("controlli", {}).get("totale_trovato_nel_documento", False))
        print(f'{d["meta"]["sorgente"][:70]:70} attese={len(gold):3} trovate={len(pred):3} esatte={ex:3}', flush=True)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    print(json.dumps({"documenti": n, "precisione_righe": round(p, 3), "richiamo_righe": round(r, 3),
                      "righe_esatte_su_attese": round(exact / (tp + fn), 3) if tp + fn else 0,
                      "documenti_perfetti": doc_ok, "documenti_con_totale_verificato": verified}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("-o", "--output")
    ap.add_argument("--eval")
    ap.add_argument("--mock", action="store_true", help="non chiama il modello (prova del flusso)")
    a = ap.parse_args()
    if a.eval:
        evaluate(a.eval, a.mock)
    elif a.file:
        res = process(a.file, a.mock)
        s = json.dumps(res, ensure_ascii=False, indent=1)
        if a.output: open(a.output, "w", encoding="utf-8").write(s)
        else: print(s)
    else:
        ap.print_help()
