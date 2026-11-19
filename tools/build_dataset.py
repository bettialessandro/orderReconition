"""
Costruisce dataset JSONL per addestrare un SLM a estrarre righe d'ordine dalle conferme fornitori.

Input : Data/ORDINI E CONFERME/<cartella brand>/**.pdf   (documenti fornitore)
Target: Data/IMPORTATI/OF_*.xlsx / RI_*.xlsx              (righe inserite dagli operatori)

Output (cartella dataset/):
  extraction_train.jsonl / extraction_test.jsonl  -> testo conferma  -> righe JSON (articolo, colore, costo, taglie)
  classify_train.jsonl   / classify_test.jsonl    -> descrizione/colore -> categoria, subcategoria, base color
  rejected.jsonl          blocchi scartati con il motivo (per revisione)
  stats.json              riepilogo
Il test set e' fatto di brand interi esclusi dal training (generalizzazione a fornitori nuovi).
"""
import os, re, json, glob, hashlib, logging, collections, warnings
import pdfplumber, openpyxl

warnings.filterwarnings("ignore")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

PRJ = r"C:\progetti\orderreconition"
SRC = os.path.join(PRJ, "Data", "ORDINI E CONFERME")
IMP = os.path.join(PRJ, "Data", "IMPORTATI")
OUT = os.path.join(PRJ, "dataset")
CACHE = os.path.join(OUT, "_pdf_text_cache")
MAX_CHARS = 8000          # dimensione massima del blocco di testo in input
TEST_BRAND_PCT = 12       # % di brand tenuti fuori per il test
MAX_NEG_PER_DOC = 3       # esempi negativi (blocchi senza righe) per documento

SYSTEM_EXTRACT = (
    "Sei un assistente che legge conferme d'ordine di fornitori moda. "
    "Estrai SOLO le righe presenti nel testo. Per ogni articolo+colore restituisci: "
    "articolo (codice fornitore), colore (codice e nome come nel documento), costo unitario, "
    "taglie (lista di {taglia, qta}) e qta totale. Rispondi solo con JSON {\"righe\": [...]}.")
SYSTEM_CLASSIFY = (
    "Classifica un articolo moda. Dati brand, genere, reparto, descrizione e colore, "
    "restituisci JSON con categoria, subcategoria e base_color usando il vocabolario aziendale.")

norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())


# ---------------------------------------------------------------- IMPORTATI
def parse_name(fname):
    """OF_W_ALAIA_PRE_ACC_23_SS26_ASIA_V3.xlsx -> metadati + chiave di gruppo (senza reparto)."""
    base = os.path.splitext(fname)[0]
    p = base.split("_")
    meta = {"file": fname, "tipo": p[0], "genere": p[1] if len(p) > 1 else "", "brand": p[2] if len(p) > 2 else ""}
    rest = p[3:]
    lines = {"PRE", "MAIN", "POST", "CORE"}
    reparti = {"RTW", "ACC", "SHOES"}
    meta["linea"] = next((x for x in rest if x.upper() in lines), "")
    meta["reparto"] = next((x for x in rest if x.upper() in reparti), "")
    # chiave di gruppo: tutto tranne reparto e suffisso versione
    meta["gruppo"] = "_".join(x for x in p[:] if x.upper() not in reparti and x.upper() not in ("V3", "PL", "ASIA"))
    meta["brand_n"] = norm(meta["brand"])
    return meta


def size_grid(rows, hdr_idx, tt_col, size_cols):
    """Righe sopra l'intestazione: col 'Tipo Taglie' contiene la lettera, a destra le etichette taglia."""
    grid = {}
    for r in rows[:hdr_idx]:
        if tt_col < len(r) and isinstance(r[tt_col], str) and len(r[tt_col].strip()) == 1:
            grid[r[tt_col].strip()] = [r[c] if c < len(r) else None for c in size_cols]
    return grid


def num(v):
    if v is None or v == "": return None
    if isinstance(v, (int, float)): return float(v)
    try: return float(str(v).replace(",", "."))
    except ValueError: return None


def read_import(path):
    """Ritorna dict (articolo, colore) -> riga normalizzata. Solleva ValueError se i fogli sono in conflitto."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    items = {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        h = next((i for i, r in enumerate(rows[:40]) if r and r[0] == "Articolo"), None)
        if h is None: continue
        hdr = [norm(x) if x is not None else "" for x in rows[h]]
        col = {}
        for i, name in enumerate(hdr[:40]):
            col.setdefault(name, i)
        size_cols = [i for i, n in enumerate(hdr[:40]) if n == "TAGLIE"]
        grid = size_grid(rows, h, col.get("TIPOTAGLIE", 8), size_cols) if size_cols else {}
        g = lambda r, k: r[col[k]] if k in col and col[k] < len(r) else None
        sheet_items = {}
        for r in rows[h + 1:]:
            if not r or not r[0] or not str(r[0]).strip(): continue
            art, colore = str(r[0]).strip(), str(g(r, "COLORE") or "").strip()
            key = (art, colore)
            it = sheet_items.setdefault(key, {
                "articolo": art, "colore": colore,
                "descrizione": g(r, "DESCRIZIONE"), "categoria": g(r, "CATEGORIA"),
                "subcategoria": g(r, "SUBCATEGORIA"), "base_color": g(r, "BASECOLOR"),
                "costo": num(g(r, "COSTO")), "retail": num(g(r, "RETAIL")),
                "materiale": g(r, "MATERIALE"), "made_in": g(r, "MADEIN"), "taglie": []})
            if size_cols:   # template V3: griglia
                labels = grid.get(str(g(r, "TIPOTAGLIE") or "").strip(), [])
                for j, c in enumerate(size_cols):
                    q = num(r[c] if c < len(r) else None)
                    if q:
                        lab = labels[j] if j < len(labels) and labels[j] is not None else f"#{j+1}"
                        it["taglie"].append({"taglia": str(lab), "qta": int(q)})
            else:           # template PL: una riga per taglia
                q = num(g(r, "QTA"))
                if q:
                    it["taglie"].append({"taglia": str(g(r, "TAGLIA") or "UNICA"), "qta": int(q)})
        for key, it in sheet_items.items():
            it["qta"] = sum(t["qta"] for t in it["taglie"])
            if it["qta"] <= 0 or not it["costo"]: continue
            if key in items and (items[key]["qta"], items[key]["costo"]) != (it["qta"], it["costo"]):
                raise ValueError(f"fogli in conflitto su {key}")
            items[key] = it
    return items


# ---------------------------------------------------------------- PDF
# due letture: con lo spazio come separatore migliaia ("1 692.00") e senza ("6 528,00" = qta 6, prezzo 528)
NUM_RES = [re.compile(r"\d{1,3}(?:[ .,'\u2019\u00a0]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"),
           re.compile(r"\d{1,3}(?:[.,'\u2019]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?")]


def count_num(text, v):
    """Quante volte l'importo v compare nel testo (lettura piu' favorevole tra le due regex)."""
    tol = max(0.05, abs(v) * 0.002)
    best = 0
    for rx in NUM_RES:
        n = sum(1 for m in rx.findall(text) if any(abs(x - v) <= tol for x in parse_numbers_raw([m])))
        best = max(best, n)
    return best


def parse_numbers(text):
    return parse_numbers_raw(x for rx in NUM_RES for x in rx.findall(text))


def parse_numbers_raw(matches):
    out = set()
    for m in matches:
        s = m.replace("\u00a0", " ")
        dec = re.search(r"[.,](\d{1,2})$", s)
        if dec and not re.search(r"[.,]\d{3}$", s):
            intpart, frac = s[:dec.start()], dec.group(1)
        else:
            intpart, frac = s, "0"
        try: out.add(round(float(re.sub(r"\D", "", intpart) + "." + frac), 2))
        except ValueError: pass
    return out


def has_num(nums, v):
    tol = max(0.05, abs(v) * 0.002)
    return any(abs(n - v) <= tol for n in nums)


def pdf_pages(path):
    key = hashlib.md5(path.encode("utf-8")).hexdigest()
    cp = os.path.join(CACHE, key + ".json")
    if os.path.exists(cp):
        return json.load(open(cp, encoding="utf-8"))
    pages = []
    try:
        with pdfplumber.open(path) as p:
            for pg in p.pages:
                pages.append(pg.extract_text() or "")
    except Exception:
        pages = []
    json.dump(pages, open(cp, "w", encoding="utf-8"), ensure_ascii=False)
    return pages


def fmt_cell(v):
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.2f}"
    if hasattr(v, "strftime"):
        return v.strftime("%d/%m/%Y")
    return str(v).strip()


def excel_pages(path, rows_per_page=60):
    """Excel fornitore -> 'pagine' di testo: una riga per riga del foglio, celle separate da ' | '."""
    key = hashlib.md5(path.encode("utf-8")).hexdigest()
    cp = os.path.join(CACHE, key + ".x.json")
    if os.path.exists(cp):
        return json.load(open(cp, encoding="utf-8"))
    pages = []
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            lines = []
            for r in ws.iter_rows(values_only=True):
                cells = [fmt_cell(c) for c in r if c is not None and str(c).strip() != ""]
                if cells: lines.append(" | ".join(cells))
                if len(lines) > 5000: break
            for i in range(0, len(lines), rows_per_page):
                pages.append(f"[Foglio: {ws.title}]\n" + "\n".join(lines[i:i + rows_per_page]))
    except Exception:
        pages = []
    json.dump(pages, open(cp, "w", encoding="utf-8"), ensure_ascii=False)
    return pages


def chunks(pages):
    cur, out = [], []
    for t in pages:
        if cur and len("\n".join(cur)) + len(t) > MAX_CHARS:
            out.append("\n".join(cur)); cur = []
        cur.append(t[:MAX_CHARS])
    if cur: out.append("\n".join(cur))
    return [c for c in out if len(c.strip()) > 50]


# ---------------------------------------------------------------- main
def main():
    os.makedirs(CACHE, exist_ok=True)
    stats = collections.Counter()

    # 1. carica IMPORTATI raggruppati per gruppo (brand+genere+linea+destinatario)
    groups = collections.defaultdict(dict)       # gruppo -> {(art,col): item}
    group_meta, by_brand = {}, collections.defaultdict(set)
    for f in sorted(os.listdir(IMP)):
        if not f.lower().endswith((".xlsx", ".xlsm")) or f.startswith("~$"): continue
        m = parse_name(f)
        try:
            items = read_import(os.path.join(IMP, f))
        except Exception as e:
            stats["import_scartati"] += 1
            print("SCARTO", f, e); continue
        stats["import_letti"] += 1
        for k, it in items.items():
            it["_reparto"] = m["reparto"]; it["_file"] = f
            groups[m["gruppo"]][k] = it
        group_meta[m["gruppo"]] = m
        by_brand[m["brand_n"]].add(m["gruppo"])

    def code_in(code, tn, up):
        c = norm(code)
        if len(c) >= 5: return c in tn
        if len(c) >= 3: return re.search(r"(?<![A-Z0-9])" + re.escape(str(code).upper()) + r"(?![A-Z0-9])", up) is not None
        return False

    def colore_doc(colore, tn):
        """Parte del colore OF leggibile nel documento; None se nemmeno il primo token (codice) c'e'."""
        toks = [t for t in re.split(r"\s+", colore.strip()) if norm(t)]
        if not toks or norm(toks[0]) not in tn: return None
        out = " ".join(t for t in toks if norm(t) in tn)
        return out if len(norm(out)) >= 2 else None

    def qty_ok(it, nums):
        return (has_num(nums, it["costo"] * it["qta"])
                or all(has_num(nums, it["costo"] * t["qta"]) for t in it["taglie"])
                or (it["qta"] > 1 and has_num(nums, it["qta"])))

    def check_rows(hits, tn, nums):
        if not all(has_num(nums, it["costo"]) for it in hits): return "costo_non_trovato"
        bad = sum(1 for it in hits if not qty_ok(it, nums))
        if bad: return f"quantita_non_verificate_{bad}_di_{len(hits)}"
        if not all(colore_doc(it["colore"], tn) for it in hits): return "colore_non_leggibile_nel_testo"
        return None

    def make_record(meta, text, tn, nums, hits):
        hits = sorted(hits, key=lambda it: (tn.find(norm(it["articolo"])), norm(it["colore"])))
        s = round(sum(it["costo"] * it["qta"] for it in hits), 2)
        meta = {**meta, "subtotale_blocco": s if len(hits) >= 2 and has_num(nums, s) else None,
                "colori_of": {f'{it["articolo"]}|{colore_doc(it["colore"], tn)}': it["colore"] for it in hits}}
        target = {"righe": [{"articolo": it["articolo"], "colore": colore_doc(it["colore"], tn),
                             "costo": it["costo"], "taglie": it["taglie"], "qta": it["qta"]} for it in hits]}
        return {"meta": meta, "messages": [
            {"role": "system", "content": SYSTEM_EXTRACT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]}

    def try_union(rej, c, full):
        """Unione delle righe di tutti i gruppi trovati nel blocco, se le chiavi non si sovrappongono."""
        if not c or len(c[3]) < 2: return None
        text, tn, nums, scored = c
        keys, union = set(), []
        for _, _, gk, hits in scored:
            for it in hits:
                k = (it["articolo"], it["colore"])
                if k in keys: return None
                keys.add(k); union.append(it)
        if check_rows(union, tn, nums): return None
        s = round(sum(it["costo"] * it["qta"] for it in union), 2)
        if has_num(nums, s): how = "subtotale_blocco_unione"
        elif count_num(full, s) >= 1: how = "totale_documento_unione"
        else: return None
        meta = {k: v for k, v in rej.items() if k != "motivo"}
        meta.update(gruppo="+".join(sorted(g for *_, g, _ in scored)), righe=len(union),
                    file_import=sorted({it["_file"] for it in union}))
        rec = make_record(meta, text, tn, nums, union)
        rec["meta"]["completezza"] = how
        if how.startswith("totale"): rec["meta"]["totale_documento"] = s
        return rec

    def accept(rec, how, tot):
        rec["meta"]["completezza"] = how
        rec["meta"]["totale_documento"] = tot
        extract.append(rec)
        stats["blocchi_sicuri_" + rec["meta"]["tipo_sorgente"]] += 1
        stats["righe_sicure"] += rec["meta"]["righe"]
        stats["completezza_" + how] += 1

    # 2. per ogni cartella fornitore -> gruppi del brand
    extract, rejected, documents = [], [], []
    for folder in sorted(os.listdir(SRC)):
        fn = norm(folder)
        cands = {gk for b, gs in by_brand.items() if b and (fn.startswith(b) or fn.startswith(b[:8])) for gk in gs}
        if not cands:
            stats["cartelle_senza_import"] += 1; continue
        sources = [p for p in glob.glob(os.path.join(SRC, folder, "**", "*"), recursive=True)
                   if p.lower().endswith((".pdf", ".xlsx", ".xlsm")) and not os.path.basename(p).startswith("~$")]
        for src in sources:
            rel = os.path.relpath(src, SRC)
            kind = "pdf" if src.lower().endswith(".pdf") else "excel"
            pages = pdf_pages(src) if kind == "pdf" else excel_pages(src)
            full = "\n".join(pages)
            pdf_nums = parse_numbers(full)
            pdf_ok, pdf_rej, pdf_rows, ctx, empty = [], [], {}, {}, []
            for ci, text in enumerate(chunks(pages)):
                stats[f"blocchi_totali_{kind}"] += 1
                tn = norm(text); up = text.upper(); nums = parse_numbers(text)
                # punteggio per gruppo
                scored = []
                for gk in cands:
                    hits = [it for (a, c), it in groups[gk].items() if code_in(a, tn, up)]
                    if not hits: continue
                    scored.append((len(hits), sum(1 for it in hits if qty_ok(it, nums)), gk, hits))
                if not scored:
                    stats["blocchi_senza_match"] += 1; empty.append((ci, text)); continue
                scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
                nh, ns, gk, hits = scored[0]
                ctx[ci] = (text, tn, nums, scored)
                reason = "ambiguo_tra_destinatari" if len(scored) > 1 and scored[1][:2] == (nh, ns) \
                    else check_rows(hits, tn, nums)
                rec_meta = {"sorgente": rel, "tipo_sorgente": kind, "blocco": ci, "gruppo": gk,
                            "brand": group_meta[gk]["brand"], "righe": len(hits),
                            "file_import": sorted({it["_file"] for it in hits})}
                for it in hits: pdf_rows[(gk, it["articolo"], it["colore"])] = it
                if reason:
                    pdf_rej.append({**rec_meta, "motivo": reason}); continue
                pdf_ok.append(make_record(rec_meta, text, tn, nums, hits))
            # completezza: totale calcolato su TUTTE le righe abbinate (anche dei blocchi scartati)
            tot = round(sum(it["costo"] * it["qta"] for it in pdf_rows.values()), 2)
            need = 1 if len(pdf_rows) >= 2 else 2      # 1 riga: importo riga + totale ordine
            completo = bool(pdf_rows) and count_num(full, tot) >= need
            for r in pdf_ok:
                if completo or r["meta"]["subtotale_blocco"]:
                    accept(r, "totale_documento" if completo else "subtotale_blocco", tot if completo else None)
                else:
                    pdf_rej.append({**r["meta"], "motivo": "totale_documento_non_coincide"})
            # recupero: blocco con piu' destinatari disgiunti, verificato per subtotale o totale
            still = []
            for r in pdf_rej:
                rec = try_union(r, ctx.get(r["blocco"]), full) if r["motivo"].startswith(("totale", "ambiguo")) else None
                if rec: accept(rec, rec["meta"]["completezza"], rec["meta"].get("totale_documento"))
                else: still.append(r)
            for r in still:
                stats["scartati_" + r["motivo"].split("_")[0]] += 1
                rejected.append(r)
            if not completo or not pdf_ok:
                continue
            brand = pdf_ok[0]["meta"]["brand"]
            # esempi negativi: blocchi di un documento completo che non contengono righe d'ordine.
            # Esclusi quelli dove compare il costo di una riga (possibile continuazione di un articolo).
            costs = {it["costo"] for it in pdf_rows.values()}
            n_neg = 0
            for ci, text in empty:
                if n_neg >= MAX_NEG_PER_DOC: break
                if any(has_num(parse_numbers(text), c) for c in costs): continue
                extract.append({"meta": {"sorgente": rel, "tipo_sorgente": kind, "blocco": ci, "brand": brand,
                                         "righe": 0, "completezza": "negativo_documento_completo"},
                                "messages": [{"role": "system", "content": SYSTEM_EXTRACT},
                                             {"role": "user", "content": text},
                                             {"role": "assistant", "content": json.dumps({"righe": []})}]})
                stats["blocchi_negativi"] += 1; n_neg += 1
            # valutazione a documento intero: solo se nessun blocco del documento e' stato scartato
            if not still and kind == "pdf":
                ftn = norm(full)
                righe = sorted(pdf_rows.values(), key=lambda it: (ftn.find(norm(it["articolo"])), norm(it["colore"])))
                documents.append({"meta": {"sorgente": rel, "brand": brand, "totale_documento": tot,
                                           "blocchi": len(chunks(pages))},
                                  "righe_attese": [{"articolo": it["articolo"], "colore": colore_doc(it["colore"], ftn),
                                                    "costo": it["costo"], "taglie": it["taglie"], "qta": it["qta"]}
                                                   for it in righe]})

    # 3. dataset classificazione (tutte le righe importate, dedup per brand+articolo+colore)
    classify, seen = [], set()
    for gk, items in groups.items():
        m = group_meta[gk]
        for (a, c), it in items.items():
            k = (m["brand_n"], a, c)
            if k in seen or not it["categoria"] or not it["descrizione"]: continue
            seen.add(k)
            inp = {"brand": m["brand"], "genere": m["genere"], "reparto": it["_reparto"],
                   "descrizione": it["descrizione"], "colore": c}
            out = {"categoria": it["categoria"], "subcategoria": it["subcategoria"], "base_color": it["base_color"]}
            classify.append({"meta": {"brand": m["brand"]}, "messages": [
                {"role": "system", "content": SYSTEM_CLASSIFY},
                {"role": "user", "content": json.dumps(inp, ensure_ascii=False)},
                {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)}]})

    # 4. split per brand
    def is_test(brand):
        return int(hashlib.md5(norm(brand).encode()).hexdigest(), 16) % 100 < TEST_BRAND_PCT

    def dump(name, recs):
        tr = [r for r in recs if not is_test(r["meta"]["brand"])]
        te = [r for r in recs if is_test(r["meta"]["brand"])]
        for suffix, data in (("train", tr), ("test", te)):
            with open(os.path.join(OUT, f"{name}_{suffix}.jsonl"), "w", encoding="utf-8") as fh:
                for r in data: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f"{name}_train"] = len(tr); stats[f"{name}_test"] = len(te)
        return sorted({r["meta"]["brand"] for r in te})

    test_brands = dump("extraction", extract)
    dump("documenti", documents)
    dump("classify", classify)
    with open(os.path.join(OUT, "rejected.jsonl"), "w", encoding="utf-8") as fh:
        for r in rejected: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    vocab = {k: sorted({json.loads(r["messages"][2]["content"])[k] or "" for r in classify} - {""})
             for k in ("categoria", "subcategoria", "base_color")}
    json.dump(vocab, open(os.path.join(OUT, "vocabolario.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    stats = dict(stats); stats["brand_test_estrazione"] = test_brands
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
