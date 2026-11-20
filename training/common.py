"""Funzioni condivise tra addestramento e valutazione."""
import json
import os
import random
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ROOT = Path(__file__).resolve().parent.parent          # radice del repository
DATA_DIR = ROOT / "dataset"
# adapter, checkpoint e valutazioni: fuori dal repository (sul volume RunPod)
RUNS_DIR = Path(os.environ.get("ORDER_RUNS_DIR", "/workspace/orderRecognition/runs"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def task_of(rec):
    """'extraction' o 'classify', dedotto dal system prompt."""
    sys = rec["messages"][0]["content"]
    return "classify" if sys.startswith("Classifica") else "extraction"


def split_by_brand(records, frac, seed):
    """Tiene da parte brand interi (come fa il test set) per la validazione."""
    brands = sorted({r["meta"].get("brand", "?") for r in records})
    rnd = random.Random(seed)
    rnd.shuffle(brands)
    n_val = max(1, round(len(brands) * frac))
    val_brands = set(brands[:n_val])
    train = [r for r in records if r["meta"].get("brand", "?") not in val_brands]
    val = [r for r in records if r["meta"].get("brand", "?") in val_brands]
    return train, val, sorted(val_brands)


def prompt_text(tok, messages):
    """Testo del prompt (system+user) con l'apertura del turno assistant."""
    return tok.apply_chat_template(
        [m for m in messages if m["role"] != "assistant"],
        tokenize=False, add_generation_prompt=True,
    )
