"""Survey: per cartella brand, quanti PDF hanno testo e quanti codici articolo degli IMPORTATI compaiono nei sorgenti."""
import os, re, sys, json, glob
import pdfplumber, openpyxl
ROOT = r"C:\progetti\orderreconition\Data"
SRC = os.path.join(ROOT, "ORDINI E CONFERME"); IMP = os.path.join(ROOT, "IMPORTATI")
norm = lambda s: re.sub(r"[^A-Z0-9]", "", s.upper())

def imp_codes(path):
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True)); hdr = None; out = []
    for i, r in enumerate(rows):
        if r and r[0] == "Articolo": hdr = i; continue
        if hdr is not None and r and r[0]: out.append(str(r[0]))
    return out

def src_text(folder):
    txt, npdf, ntext = [], 0, 0
    for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        ext = f.lower().rsplit(".", 1)[-1]
        try:
            if ext == "pdf":
                npdf += 1
                with pdfplumber.open(f) as p:
                    t = "\n".join((pg.extract_text() or "") for pg in p.pages[:30])
                if len(t.strip()) > 50: ntext += 1
                txt.append(t)
            elif ext in ("xlsx", "xlsm"):
                wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
                for ws in wb.worksheets:
                    for r in ws.iter_rows(values_only=True):
                        txt.append(" ".join(str(c) for c in r if c is not None))
        except Exception as e:
            pass
    return norm(" ".join(txt)), npdf, ntext

imps = os.listdir(IMP)
res = []
for d in sorted(os.listdir(SRC)):
    brand = norm(re.split(r" (Main|Pre|Post|M Main|Core)\b", d)[0])
    files = [f for f in imps if f.endswith(".xlsx") and brand[:8] in norm(f)]
    if not files: res.append((d, 0, 0, 0, 0, 0)); continue
    codes = set()
    for f in files:
        try: codes |= set(imp_codes(os.path.join(IMP, f)))
        except Exception: pass
    blob, npdf, ntext = src_text(os.path.join(SRC, d))
    hit = sum(1 for c in codes if norm(c) and norm(c) in blob)
    res.append((d, len(files), len(codes), hit, npdf, ntext))
    print(f"{d[:45]:45} imp={len(files):2} codes={len(codes):4} found={hit:4} pdf={npdf:3} pdf_txt={ntext:3}", flush=True)
json.dump(res, open(os.path.join(ROOT, "..", "tools", "survey.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
