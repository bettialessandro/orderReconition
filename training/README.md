# Addestramento SLM (Qwen3-4B-Instruct + LoRA)

Script per addestrare e valutare il modello sul dataset in `../dataset`. Si lanciano dalla radice del repository.
Adapter, checkpoint e valutazioni vanno in `$ORDER_RUNS_DIR` (default `/workspace/orderRecognition/runs`, volume RunPod), fuori dal repository.

| Script | Uso |
|---|---|
| `train.py` | LoRA bf16 r=32 su tutti i layer lineari, 3 epoche, max 8192 token, validazione su brand esclusi |
| `evaluate_vllm.py` | valutazione sui test set con vLLM (consigliata: ~2 min) |
| `evaluate.py` | stessa valutazione con transformers (lenta su output lunghi); contiene le metriche |
| `rescore.py` | ricalcola le metriche dalle predizioni salvate |
| `merge.py` | fonde l'adapter nel modello base per il serving |

```
pip install "transformers>=4.56,<5" "peft>=0.17" "accelerate>=1.10" "datasets>=4.0"
python training/train.py --run_name qwen3-4b-r32

# vLLM in un ambiente separato (porta con sé un'altra versione di torch)
python3 -m venv /root/vllm-env && /root/vllm-env/bin/pip install vllm
/root/vllm-env/bin/python training/evaluate_vllm.py --run qwen3-4b-r32
/root/vllm-env/bin/python training/evaluate_vllm.py --base_only
```
Requisiti misurati (RTX PRO 4500 Blackwell 32 GB): picco 25,2 GB, 60 min di addestramento.

## Risultati `qwen3-4b-r32` (test: 16 brand mai visti)
| Metrica | Base | Addestrato |
|---|---|---|
| Righe F1 (colore tollerante) | 39% | 76% |
| Costo / quantità corretti | 0% / 0% | 99,6% / 97,5% |
| Taglie corrette | 7% | 50% |
| Blocchi positivi esatti | 0% | 27% |
| Classificazione: categoria / subcategoria / base color | 0% / 3% / 12% | 87% / 76% / 77% |

Il 78% degli errori sulle taglie nasce dall'estrazione del testo, che perde le celle vuote della griglia taglie (`excel_pages` scarta le celle vuote, `pdfplumber.extract_text` non conserva le colonne).
