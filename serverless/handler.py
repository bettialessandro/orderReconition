"""
Worker RunPod Serverless: lettura conferme d'ordine fornitore con il modello Qwen addestrato.

Il modello NON e' nell'immagine: viene caricato dal network volume, montato dai worker
serverless in /runpod-volume (sul Pod lo stesso volume e' /workspace).

Input (job["input"]), una delle forme:

  1. Documento intero (PDF o Excel .xlsx), stessa pipeline di tools/infer.py:
       {"file_base64": "<...>", "filename": "conferma.pdf"}
       {"file_url": "https://.../conferma.pdf", "filename": "conferma.pdf"}   # filename opzionale
     -> {"file", "righe": [...], "controlli": {...}}

  2. Un blocco di testo gia' estratto (come negli esempi del dataset):
       {"text": "<testo della conferma>"}
     -> {"righe": [...], "controlli": {"righe_taglie_sospette": [...]}}

  3. Classificazione articoli (uno o una lista):
       {"classify": {"brand": "...", "genere": "...", "reparto": "...", "descrizione": "...", "colore": "..."}}
     -> {"risultati": [{"categoria", "subcategoria", "base_color"} | {"errore"}]}

Variabili d'ambiente: vedi serverless/README.md.
"""
import base64
import json
import os
import sys
import tempfile
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import runpod  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

import build_dataset as B  # noqa: E402
import infer  # noqa: E402
from controlli import controlla_taglie  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/orderRecognition/runs/qwen3-4b-r32/merged")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "0") == "1"
MAX_FILE_MB = float(os.environ.get("MAX_FILE_MB", "50"))
CLASSIFY_FIELDS = ("brand", "genere", "reparto", "descrizione", "colore")


def _load_model():
    if not os.path.isfile(os.path.join(MODEL_PATH, "config.json")):
        raise RuntimeError(
            f"Modello non trovato in {MODEL_PATH}. Controlla che il network volume sia collegato "
            f"all'endpoint (montato in /runpod-volume) e la variabile MODEL_PATH.")
    t0 = time.time()
    llm = LLM(MODEL_PATH, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=GPU_MEMORY_UTILIZATION, enforce_eager=ENFORCE_EAGER, seed=0)
    print(f"[worker] modello caricato da {MODEL_PATH} in {time.time() - t0:.0f}s", flush=True)
    return llm


# caricato una volta all'avvio del worker, riusato da tutte le richieste
LLM_ENGINE = _load_model()


def _chat(system, user_texts, max_tokens):
    convs = [[{"role": "system", "content": system}, {"role": "user", "content": u}] for u in user_texts]
    outs = LLM_ENGINE.chat(convs, SamplingParams(temperature=0, max_tokens=max_tokens), use_tqdm=False)
    return [o.outputs[0].text for o in outs]


def extract_many(blocks):
    """Lista di blocchi di testo -> lista di {"righe": [...]} (generazione in un solo batch)."""
    if not blocks:
        return []
    return [infer.parse_output(t) for t in _chat(B.SYSTEM_EXTRACT, blocks, MAX_TOKENS)]


# --------------------------------------------------------------------------- modalita'
def run_document(inp):
    name = os.path.basename(inp.get("filename") or "")
    if inp.get("file_base64"):
        try:
            data = base64.b64decode(inp["file_base64"], validate=True)
        except Exception:
            return {"error": "file_base64 non e' base64 valido"}
    else:
        url = inp["file_url"]
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            return {"error": "file_url deve essere http(s)"}
        name = name or os.path.basename(urllib.parse.urlparse(url).path)
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read(int(MAX_FILE_MB * 1024 * 1024) + 1)
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return {"error": f"file oltre {MAX_FILE_MB:.0f} MB"}

    ext = os.path.splitext(name)[1].lower()
    if ext not in (".pdf", ".xlsx", ".xlsm"):
        if data[:4] == b"%PDF":
            ext = ".pdf"
        else:
            return {"error": "formato non supportato: servono PDF o Excel .xlsx (indicare 'filename')"}

    with tempfile.TemporaryDirectory() as tmp:
        # build_dataset mette in cache il testo estratto in B.CACHE: una cartella per richiesta
        B.CACHE = tmp
        path = os.path.join(tmp, "documento" + ext)
        with open(path, "wb") as f:
            f.write(data)
        res = infer.process(path, extract_many=extract_many)
    res["file"] = name or ("documento" + ext)
    return res


def run_text(inp):
    out = extract_many([inp["text"]])[0]
    nums = B.parse_numbers(inp["text"])
    sospette = []
    for r in out["righe"]:
        problemi = controlla_taglie(r)
        if not infer.quantita_verificata(r, nums):
            problemi.append("quantita_non_trovata_nel_documento")
        if problemi:
            sospette.append({"articolo": r.get("articolo"), "colore": r.get("colore"), "problemi": problemi})
    res = {"righe": out["righe"], "controlli": {"righe_taglie_sospette": sospette,
                                                "da_verificare": bool(sospette or out.get("errore"))}}
    if out.get("errore"):
        res["errore"] = out["errore"]
    return res


def run_classify(inp):
    items = inp["classify"] if isinstance(inp["classify"], list) else [inp["classify"]]
    if not items or not all(isinstance(x, dict) for x in items):
        return {"error": "classify: serve un oggetto o una lista di oggetti"}
    # stesso formato degli esempi di addestramento: chiavi in quest'ordine, JSON non-ASCII
    users = [json.dumps({k: x.get(k, "") for k in CLASSIFY_FIELDS}, ensure_ascii=False) for x in items]
    risultati = []
    for t in _chat(B.SYSTEM_CLASSIFY, users, 128):
        try:
            risultati.append(json.loads(t))
        except json.JSONDecodeError:
            risultati.append({"errore": "json_non_valido", "raw": t[:300]})
    return {"risultati": risultati}


def handler(job):
    inp = job.get("input") or {}
    t0 = time.time()
    try:
        if inp.get("file_base64") or inp.get("file_url"):
            res = run_document(inp)
        elif isinstance(inp.get("text"), str) and inp["text"].strip():
            res = run_text(inp)
        elif inp.get("classify"):
            res = run_classify(inp)
        else:
            return {"error": "input non valido: usare file_base64 / file_url, text oppure classify"}
    except Exception as e:  # l'errore torna al chiamante invece di far cadere il worker
        return {"error": f"{type(e).__name__}: {e}"}
    if isinstance(res, dict) and "error" not in res:
        res["secondi"] = round(time.time() - t0, 2)
    return res


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
