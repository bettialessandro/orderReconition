# Worker RunPod Serverless

Endpoint serverless che legge le conferme d'ordine con il modello Qwen3-4B addestrato
(`training/`, run `qwen3-4b-r32`). Riusa la pipeline di `tools/infer.py`: estrazione testo →
blocchi → modello (vLLM, tutti i blocchi in un batch) → unione righe → controlli
(totale documento, costi, taglie).

## Dove sta il modello

Il modello **non è nell'immagine**: si carica dal network volume `net_finetuning_order`.

| | Pod | Serverless |
|---|---|---|
| punto di montaggio del volume | `/workspace` | `/runpod-volume` |
| modello (adapter già fuso) | `/workspace/orderRecognition/runs/qwen3-4b-r32/merged` | `/runpod-volume/orderRecognition/runs/qwen3-4b-r32/merged` |

Per un nuovo modello: `python training/merge.py --run <nome>` su un Pod, poi cambia la variabile
`MODEL_PATH` dell'endpoint (nessuna nuova build).

## Deploy da GitHub

1. **Serverless → New Endpoint → GitHub Repo**, scegli questo repository e il branch.
   Dockerfile Path: `Dockerfile` (nella radice). Il `.dockerignore` lascia fuori `dataset/`.
2. **Endpoint Type**: Queue.
3. **GPU**: 24 GB o più (L4, A5000, RTX 4090, L40S…). Il modello occupa 7,6 GB; il resto è cache KV.
   Testato su RTX PRO 4500 Blackwell 32 GB.
4. **Network Volume**: seleziona `net_finetuning_order`. I worker vengono creati solo nel
   datacenter del volume (EU-RO-1): scegli GPU disponibili lì.
5. **Execution Timeout**: almeno 600 s (un documento lungo ha molti blocchi).
6. **Variabili d'ambiente** (tutte opzionali, i default sono nel Dockerfile):

   | Variabile | Default | Note |
   |---|---|---|
   | `MODEL_PATH` | `/runpod-volume/orderRecognition/runs/qwen3-4b-r32/merged` | cartella del modello sul volume |
   | `VLLM_CACHE_ROOT` | `/runpod-volume/vllm-cache` | cache di compilazione condivisa tra i worker |
   | `MAX_MODEL_LEN` | `16384` | blocco ≤ 8000 caratteri + risposta |
   | `MAX_TOKENS` | `8192` | lunghezza massima della risposta per blocco |
   | `GPU_MEMORY_UTILIZATION` | `0.90` | |
   | `ENFORCE_EAGER` | `0` | `1` = avvio più rapido senza cache, ma generazione ~2× più lenta |
   | `MAX_FILE_MB` | `50` | dimensione massima del file |

Le modifiche al repository non si applicano da sole: per ridistribuire crea una nuova **release** su GitHub.

## Tempi misurati (RTX PRO 4500, vLLM 0.32)

| | |
|---|---|
| avvio worker, prima volta su un tipo di GPU (compilazione) | ~230 s |
| avvio worker con la cache di compilazione sul volume | ~65 s (di cui ~25 s lettura pesi dal volume) |
| avvio con `ENFORCE_EAGER=1` | ~50 s |
| blocco di testo (12 righe) | ~14 s (~27 s in eager) |

Per evitare l'attesa sul primo job imposta **Active Workers = 1** (worker sempre acceso, a pagamento)
oppure attiva FlashBoot.

## Chiamate

Input (`"input"` del job), una delle tre forme:

```jsonc
// 1. documento intero: PDF o Excel .xlsx (in base64 oppure da URL)
{"input": {"file_base64": "<base64>", "filename": "conferma.pdf"}}
{"input": {"file_url": "https://…/conferma.pdf"}}

// 2. un blocco di testo già estratto
{"input": {"text": "Ordine 104.446 …"}}

// 3. classificazione articoli (oggetto o lista)
{"input": {"classify": [{"brand": "GOLDENGOOSE", "genere": "K", "reparto": "SHOES",
                         "descrizione": "'Super-Star' Sneakers", "colore": "WHITE/ICE/NAVY BLUE"}]}}
```

Esempio con un PDF:

```bash
B64=$(base64 -w0 conferma.pdf)
curl -s https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d "{\"input\": {\"file_base64\": \"$B64\", \"filename\": \"conferma.pdf\"}}"
```

Per documenti lunghi usa `/run` (asincrono) e poi `/status/<id>`. Il payload della richiesta ha un
limite di dimensione: per file grandi usa `file_url`.

Risposta per un documento:

```jsonc
{"file": "conferma.pdf",
 "righe": [{"articolo": "266200", "colore": "CREAM BOTANIC", "costo": 116.0,
            "taglie": [{"taglia": "XS", "qta": 1}, …], "qta": 6}],
 "controlli": {"blocchi": 1, "errori_blocchi": [], "totale_righe": 976.0,
               "totale_trovato_nel_documento": true, "righe_con_costo_non_trovato": [],
               "righe_taglie_sospette": [], "da_verificare": false},
 "secondi": 1.9}
```

`controlli.da_verificare = true` significa che il risultato va controllato da un operatore.
In caso di input non valido o errore la risposta è `{"error": "..."}`.

## Limiti noti

- Le taglie di una griglia con celle vuote possono essere assegnate alla colonna sbagliata
  (l'estrazione del testo perde le celle vuote): la quantità totale resta giusta e il controllo
  non lo rileva. Vedi `training/README.md`.
- PDF scansionati (senza testo) non sono supportati: la risposta lo segnala.
- File `.xls` non supportati.

## Prova in locale (su un Pod con GPU)

```bash
pip install vllm==0.32.0 runpod==1.10.0 pdfplumber==0.11.10 openpyxl==3.1.5
MODEL_PATH=/workspace/orderRecognition/runs/qwen3-4b-r32/merged \
  python serverless/handler.py --test_input "$(cat serverless/test_input.json)"
```
