"""
Controlli sulle righe estratte dal modello, senza dipendenze esterne.

- abbina_righe: abbina righe attese e righe estratte per articolo + colore, tollerando
  colori scritti in modo diverso (es. "1 BLUE SEALACE" vs "BLUE SEALACE").
- controlla_taglie: segnala le righe con taglie incoerenti o malformate.

Usato da infer.py (produzione e --eval) e dagli script di valutazione del modello.
"""
import re

norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())   # come build_dataset.norm

# taglie "lettera" per adulti: servono a riconoscere la quantita' attaccata alla taglia ("2M")
_TAGLIE_LETTERA = r"(?:XXXS|XXS|XS|S|M|L|XXL|XXXL)"
_TAGLIA_CON_QTA = re.compile(r"^(\d{1,2})" + _TAGLIE_LETTERA + r"$")


# --------------------------------------------------------------------------- colore
def token_colore(c):
    """Token normalizzati del colore, divisi come in build_dataset.colore_doc (per spazi)."""
    return {norm(t) for t in re.split(r"\s+", str(c or "").strip()) if norm(t)}


def colori_compatibili(a, b):
    """True se i due colori coincidono o se i token di uno sono contenuti in quelli dell'altro.
    Il documento puo' riportare token in piu' (prefissi, codici) o in meno rispetto all'OF."""
    ta, tb = token_colore(a), token_colore(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    piccolo, grande = sorted((ta, tb), key=len)
    return piccolo <= grande and len("".join(piccolo)) >= 3


def abbina_righe(attese, estratte):
    """Abbinamento uno-a-uno tra righe attese ed estratte.

    Passo 1: articolo e colore identici (normalizzati).
    Passo 2: stesso articolo e colori compatibili, preferendo la maggiore sovrapposizione
             di token; ogni riga si usa una volta sola, cosi' "BLUE" e "BLUE SEALACE" dello
             stesso articolo non finiscono sulla stessa riga.

    Ritorna (coppie [(attesa, estratta, esatto: bool)], attese_non_trovate, estratte_in_piu).
    """
    libere_a = list(range(len(attese)))
    libere_e = list(range(len(estratte)))
    coppie = []

    chiave = lambda r: (norm(r.get("articolo")), norm(r.get("colore")))
    per_chiave = {}
    for j in libere_e:
        per_chiave.setdefault(chiave(estratte[j]), []).append(j)
    for i in list(libere_a):
        cand = per_chiave.get(chiave(attese[i]))
        if cand:
            j = cand.pop(0)
            coppie.append((attese[i], estratte[j], True))
            libere_a.remove(i); libere_e.remove(j)

    candidati = []
    for i in libere_a:
        ta = token_colore(attese[i].get("colore"))
        for j in libere_e:
            if (norm(attese[i].get("articolo")) == norm(estratte[j].get("articolo"))
                    and colori_compatibili(attese[i].get("colore"), estratte[j].get("colore"))):
                te = token_colore(estratte[j].get("colore"))
                candidati.append((len(ta & te) / len(ta | te), i, j))
    for _, i, j in sorted(candidati, key=lambda x: -x[0]):
        if i in libere_a and j in libere_e:
            coppie.append((attese[i], estratte[j], False))
            libere_a.remove(i); libere_e.remove(j)

    return coppie, [attese[i] for i in libere_a], [estratte[j] for j in libere_e]


# --------------------------------------------------------------------------- taglie
def _qta(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def controlla_taglie(riga):
    """Lista dei problemi della riga sulle taglie (vuota se tutto torna).

    - qta_taglia_non_valida:      quantita' per taglia mancante, non numerica o <= 0
    - taglie_duplicate:           la stessa taglia compare due volte
    - somma_taglie_diversa_da_qta: la somma delle quantita' per taglia non coincide con qta
    - quantita_attaccata_alla_taglia: taglie come "1S", "2M" (la quantita' letta come parte
      della taglia). Si segnala solo se TUTTE le taglie della riga hanno questa forma e non
      sono solo "<n>M", per non confondere le taglie bimbo in mesi (6M, 12M...).
    """
    problemi = []
    taglie = riga.get("taglie") or []
    if not taglie:
        return problemi

    qta_taglie = [_qta(t.get("qta")) for t in taglie]
    if any(q is None or q <= 0 for q in qta_taglie):
        problemi.append("qta_taglia_non_valida")

    nomi = [norm(t.get("taglia")) for t in taglie]
    if len(set(nomi)) != len(nomi):
        problemi.append("taglie_duplicate")

    qta = _qta(riga.get("qta"))
    if qta is not None and all(q is not None for q in qta_taglie) and abs(sum(qta_taglie) - qta) > 1e-6:
        problemi.append("somma_taglie_diversa_da_qta")

    if nomi and all(_TAGLIA_CON_QTA.match(n) for n in nomi) and not all(n.endswith("M") for n in nomi):
        problemi.append("quantita_attaccata_alla_taglia")

    return problemi
