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
from controlli import abbina_righe, controlla_taglie

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
    # controllo taglie per blocco, PRIMA dell'unione: merge_rows ricalcola qta dalle taglie
    # e nasconderebbe le incoerenze tra quantita' totale e quantita' per taglia
    taglie_sospette = []
    for i, (o, t) in enumerate(zip(outs, blocks)):
        nums_blocco = B.parse_numbers(t)
        for r in o["righe"]:
            problemi = controlla_taglie(r)
            if not quantita_verificata(r, nums_blocco):
                problemi.append("quantita_non_trovata_nel_documento")
            if problemi:
                taglie_sospette.append({"blocco": i, "articolo": r.get("articolo"),
                                        "colore": r.get("colore"), "problemi": problemi})
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
        "righe_taglie_sospette": taglie_sospette,
    }
    controlli["da_verificare"] = (not controlli["totale_trovato_nel_documento"]
                                  or bool(controlli["errori_blocchi"])
                                  or bool(controlli["righe_con_costo_non_trovato"])
                                  or bool(taglie_sospette))
    return {"file": path, "righe": righe, "controlli": controlli}


def quantita_verificata(r, nums):
    """Stesso criterio di build_dataset.qty_ok: la quantita' e' confermata dal documento se ne
    compare l'importo riga, l'importo di ogni taglia o (se > 1) la quantita' totale."""
    try:
        costo, qta = float(r.get("costo")), float(r.get("qta"))
        q_taglie = [float(t["qta"]) for t in r.get("taglie") or []]
    except (TypeError, ValueError, KeyError):
        return False
    return (B.has_num(nums, costo * qta)
            or (bool(q_taglie) and all(B.has_num(nums, costo * q) for q in q_taglie))
            or (qta > 1 and B.has_num(nums, qta)))


def evaluate(jsonl, mock=False):
    """Abbinamento righe con colore tollerante (controlli.abbina_righe); riporta anche quante
    coppie hanno il colore identico, per confronto con la misura rigida."""
    def sizes(r):
        try:
            return sorted((B.norm(t["taglia"]), float(t["qta"])) for t in r.get("taglie") or [])
        except (TypeError, ValueError, KeyError):
            return None
    tp = tp_rigido = fp = fn = exact = doc_ok = verified = segnalati = n = 0
    for line in open(jsonl, encoding="utf-8"):
        d = json.loads(line); n += 1
        res = process(os.path.join(B.SRC, d["meta"]["sorgente"]), mock)
        coppie, mancanti, in_piu = abbina_righe(d["righe_attese"], res["righe"])
        tp += len(coppie); tp_rigido += sum(1 for *_, esatto in coppie if esatto)
        fp += len(in_piu); fn += len(mancanti)
        ex = sum(1 for g, p, _ in coppie if abs((p.get("costo") or 0) - g["costo"]) < 0.01
                 and sizes(p) == sizes(g))
        exact += ex
        doc_ok += int(ex == len(d["righe_attese"]) == len(res["righe"]))
        verified += int(res.get("controlli", {}).get("totale_trovato_nel_documento", False))
        segnalati += int(res.get("controlli", {}).get("da_verificare", False))
        print(f'{d["meta"]["sorgente"][:70]:70} attese={len(d["righe_attese"]):3} '
              f'trovate={len(res["righe"]):3} esatte={ex:3}', flush=True)
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    print(json.dumps({"documenti": n, "precisione_righe": round(p, 3), "richiamo_righe": round(r, 3),
                      "righe_abbinate_con_colore_identico": tp_rigido, "righe_abbinate": tp,
                      "righe_esatte_su_attese": round(exact / (tp + fn), 3) if tp + fn else 0,
                      "documenti_perfetti": doc_ok, "documenti_con_totale_verificato": verified,
                      "documenti_da_verificare": segnalati}, indent=1))

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
