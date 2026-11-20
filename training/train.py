"""Fine-tuning LoRA (bf16) di Qwen3-4B-Instruct sulle conferme d'ordine.

Un solo modello per i due compiti (estrazione righe + classificazione articolo),
distinti dal system prompt. La loss è calcolata solo sulla risposta dell'assistant.

Esempio:
    python training/train.py --run_name qwen3-4b-r32
    python training/train.py --max_steps 5 --run_name smoke     # prova veloce
"""
import argparse
import json
import random
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from common import (DATA_DIR, DEFAULT_MODEL, RUNS_DIR, prompt_text, read_jsonl,
                    split_by_brand)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--run_name", default=None)
    p.add_argument("--max_len", type=int, default=8192,
                   help="esempi più lunghi vengono scartati (e contati nel log)")
    p.add_argument("--classify_n", type=int, default=1500,
                   help="esempi di classificazione campionati (0 = nessuno, -1 = tutti)")
    p.add_argument("--extraction_repeat", type=int, default=1,
                   help="ripete gli esempi di estrazione N volte nel train")
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="quota di brand del train tenuti per la validazione")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def encode(tok, rec, max_len):
    msgs = rec["messages"]
    prompt = prompt_text(tok, msgs)
    answer = msgs[-1]["content"] + tok.eos_token + "\n"   # <|im_end|>\n
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    ids = p_ids + a_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + a_ids}


def build(tok, records, max_len, name):
    out, too_long = [], 0
    for r in records:
        e = encode(tok, r, max_len)
        if e is None:
            too_long += 1
        else:
            out.append(e)
    lens = sorted(len(e["input_ids"]) for e in out)
    print(f"[{name}] esempi={len(out)} scartati_troppo_lunghi={too_long} "
          f"token med={lens[len(lens)//2]} max={lens[-1]}")
    return Dataset.from_list(out), too_long


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id)
        lab = torch.full((len(batch), n), -100)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            lab[i, :k] = torch.tensor(b["labels"])
            att[i, :k] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return fn


def main():
    a = parse_args()
    random.seed(a.seed)
    run_name = a.run_name or time.strftime("run-%Y%m%d-%H%M%S")
    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- dati -------------------------------------------------------------
    ext = read_jsonl(DATA_DIR / "extraction_train.jsonl")
    ext_tr, ext_val, val_brands = split_by_brand(ext, a.val_frac, a.seed)

    cls = read_jsonl(DATA_DIR / "classify_train.jsonl")
    random.shuffle(cls)
    cls_val, cls = cls[:200], cls[200:]
    if a.classify_n >= 0:
        cls = cls[:a.classify_n]

    train_recs = ext_tr * a.extraction_repeat + cls
    random.shuffle(train_recs)
    val_recs = ext_val + cls_val[:100]

    train_ds, tl_tr = build(tok, train_recs, a.max_len, "train")
    val_ds, tl_val = build(tok, val_recs, a.max_len, "val")
    print(f"brand di validazione: {val_brands}")

    # --- modello ----------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch" if a.max_steps < 0 else "no",
        save_strategy="epoch" if a.max_steps < 0 else "no",
        save_total_limit=2,
        load_best_model_at_end=a.max_steps < 0,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=a.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collate(tok.pad_token_id))
    t0 = time.time()
    trainer.train()
    final_eval = trainer.evaluate()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    summary = {
        "args": vars(a), "val_brands": val_brands,
        "n_train": len(train_ds), "n_val": len(val_ds),
        "scartati_troppo_lunghi": {"train": tl_tr, "val": tl_val},
        "eval": final_eval, "minuti": round((time.time() - t0) / 60, 1),
        "picco_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        "log": trainer.state.log_history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nAdapter salvato in {adapter_dir}  (eval_loss={final_eval.get('eval_loss'):.4f}, "
          f"picco VRAM {summary['picco_vram_gb']} GB, {summary['minuti']} min)")


if __name__ == "__main__":
    main()
