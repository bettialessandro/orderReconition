"""Fonde l'adapter LoRA nel modello base, per servirlo con vLLM / llama.cpp.

    python training/merge.py --run qwen3-4b-r32
    vllm serve /workspace/orderRecognition/runs/qwen3-4b-r32/merged --max-model-len 16384
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DEFAULT_MODEL, RUNS_DIR

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--model", default=DEFAULT_MODEL)
a = p.parse_args()

run = RUNS_DIR / a.run
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, run / "adapter").merge_and_unload()
model.save_pretrained(run / "merged", safe_serialization=True)
AutoTokenizer.from_pretrained(run / "adapter").save_pretrained(run / "merged")
print(f"Modello fuso in {run / 'merged'}")
