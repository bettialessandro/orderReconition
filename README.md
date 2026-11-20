# orderReconition

Addestramento di un modello linguistico piccolo (SLM) che legge le conferme d'ordine dei brand moda (PDF/Excel) e restituisce le righe d'ordine nel formato degli OF importati.

> ⚠️ Repository **privato**: `dataset/` contiene costi d'acquisto e dati commerciali di fornitori (IBAN, P.IVA, email). Non renderlo pubblico.

## Struttura
| Percorso | Contenuto |
|---|---|
| `tools/build_dataset.py` | genera il dataset JSONL abbinando i documenti fornitore (`Data/ORDINI E CONFERME`) agli OF inseriti dagli operatori (`Data/IMPORTATI`) |
| `tools/infer.py` | pipeline di produzione: PDF → righe JSON con verifica del totale; `--eval` per la valutazione a documento intero |
| `tools/survey.py` | analisi preliminare di copertura sorgenti ↔ IMPORTATI |
| `dataset/` | dataset generato (vedi `dataset/README.md`) |
| `tools/controlli.py` | abbinamento righe con colore tollerante e controlli sulle taglie |
| `training/` | addestramento e valutazione del modello (vedi `training/README.md`) |
| `serverless/`, `Dockerfile` | worker RunPod Serverless con il modello sul network volume (vedi `serverless/README.md`) |

`Data/` non è nel repository: per rigenerare il dataset va copiata in locale nella radice del progetto.

## Uso
```
pip install pdfplumber openpyxl
python tools/build_dataset.py
python tools/infer.py conferma.pdf -o risultato.json
python tools/infer.py --eval dataset/documenti_test.jsonl
```
