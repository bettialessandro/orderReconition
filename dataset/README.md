# Dataset addestramento SLM – conferme d'ordine fornitori

Generato da `python tools/build_dataset.py` (il testo dei PDF è in cache in `_pdf_text_cache/`, i giri successivi sono rapidi).

## File
| File | Contenuto |
|---|---|
| `extraction_train/test.jsonl` | testo di un blocco di conferma PDF (≤ 8000 caratteri) → `{"righe":[{articolo, colore, costo, taglie:[{taglia,qta}], qta}]}` |
| (dentro `extraction_*`) | **esempi negativi** (`meta.completezza = negativo_documento_completo`): pagine senza righe d'ordine (condizioni, composizioni…) di documenti verificati → `{"righe": []}` |
| `documenti_train/test.jsonl` | valutazione **a documento intero**: percorso del PDF + tutte le righe attese. Solo documenti verificati in cui nessun blocco è stato scartato. Usare con `python tools/infer.py --eval` |
| `classify_train/test.jsonl` | `{brand, genere, reparto, descrizione, colore}` → `{categoria, subcategoria, base_color}` |
| `vocabolario.json` | valori ammessi per categoria / subcategoria / base color |
| `rejected.jsonl` | blocchi scartati con il motivo, da rivedere o recuperare |
| `stats.json` | riepilogo numerico |

Ogni riga JSONL ha `messages` (system/user/assistant, formato chat standard per il fine-tuning SFT) e `meta` (PDF di origine, file IMPORTATI, gruppo, ecc. — non va passato al modello).
Il test set contiene **brand interi** esclusi dal training (misura la generalizzazione a fornitori nuovi).

## In produzione
`python tools/infer.py conferma.pdf` → estrae il testo, lo divide in blocchi **con le stesse funzioni del dataset**, chiama il modello per blocco (endpoint OpenAI-compatibile locale: `SLM_URL`, `SLM_MODEL`), unisce le righe e verifica che la somma coincida con un totale del documento (`controlli.da_verificare`).
I file `*_test` non vanno mai usati per l'addestramento: servono a misurare il modello su brand che non ha visto.

## Criteri "coppia sicura" (estrazione)
Un blocco entra nel dataset solo se:
1. il codice di ogni riga target compare nel testo (codici < 5 caratteri: solo a parola intera);
2. il blocco è attribuibile a un solo gruppo IMPORTATI (brand + genere + linea + destinatario), altrimenti è `ambiguo`;
3. il costo unitario di ogni riga compare nel testo;
4. la quantità è verificata (importo riga, importi per taglia o quantità totale);
5. il codice colore è leggibile nel testo; nel target resta **solo la parte di colore presente nel documento** (il colore completo dell'operatore è in `meta.colori_of`);
6. **completezza** (`meta.completezza`), una delle due:
   - `totale_documento`: la somma costo×qta di tutte le righe abbinate nel documento coincide con un importo del documento; se la riga è una sola, l'importo deve comparire almeno due volte (riga + totale ordine);
   - `subtotale_blocco`: il blocco ha ≥ 2 righe e la loro somma compare nel blocco (subtotale di sezione).
   Questo evita esempi in cui il modello impara a saltare righe.
7. recupero `*_unione`: se un blocco contiene righe di più destinatari con chiavi disgiunte, si usa l'unione, purché superi gli stessi controlli.

Sorgenti: PDF e Excel dei fornitori (`meta.tipo_sorgente`). Gli Excel sono serializzati riga per riga con celle separate da ` | ` e pagine di 60 righe.
Fuori ambito: PDF scansionati (≈1%), file .xls, retail/descrizione/materiale (provengono da listini e linesheet, non dalla conferma).
Documenti con righe non presenti negli IMPORTATI (cancellazioni, ordini non definitivi) restano esclusi di proposito.
