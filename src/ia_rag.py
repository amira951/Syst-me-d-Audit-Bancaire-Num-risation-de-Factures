"""
Embedding : texte → vecteur de nombres.
Similarité cosine : cos(θ) entre 2 vecteurs = proche de 1 si même SENS.

RAG (Retrieval-Augmented Generation) :
  1. on embarque la question
  2. on récupère les K factures les plus proches
  3. on donne SEULEMENT ça au LLM
→ moins d'hallucinations, et tu peux CITER tes sources.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

try:
    import ollama
except ImportError:
    ollama = None

from src.llm_extraction import appeler_llm_texte, ExtractionLLMError

MODELE_EMBED = "nomic-embed-text"


def _nettoyer(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def embed(texte: str) -> np.ndarray | None:
    if ollama is None:
        return None
    texte = _nettoyer(texte) or " "
    try:
        if hasattr(ollama, "embed"):
            r = ollama.embed(model=MODELE_EMBED, input=texte)
            vecs = r["embeddings"] if isinstance(r, dict) else r.embeddings
            return np.array(vecs[0], dtype=np.float32)
        r = ollama.embeddings(model=MODELE_EMBED, prompt=texte)
        vec = r["embedding"] if isinstance(r, dict) else r.embedding
        return np.array(vec, dtype=np.float32)
    except Exception:
        return None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _texte_facture(f: dict) -> str:
    bits = [
        f.get("invoice_no") or f.get("numero") or "",
        f.get("seller_name") or f.get("vendeur") or "",
        f.get("client_name") or f.get("client") or "",
        str(f.get("gross_worth_total") or f.get("total") or ""),
        f.get("date") or "",
        f.get("nom_fichier") or f.get("fichier") or "",
    ]
    return _nettoyer(" | ".join(str(b) for b in bits if b))


def _matrice(factures: list[dict]) -> tuple[np.ndarray | None, list[str]]:
    vecs, textes = [], []
    for f in factures:
        t = _texte_facture(f)
        v = embed(t)
        textes.append(t)
        if v is None:
            return None, textes
        vecs.append(v)
    return np.vstack(vecs), textes


def factures_similaires(courante: dict, historique: list[dict], k: int = 3) -> list[dict]:
    if not historique:
        return []
    v0 = embed(_texte_facture(courante))
    M, _ = _matrice(historique)
    if v0 is None or M is None:
        return _similaires_tfidf(courante, historique, k)

    scores = [cosine(v0, M[i]) for i in range(len(historique))]
    ordre = np.argsort(scores)[::-1][:k]
    return [
        {
            "rang": i + 1,
            "score": round(float(scores[j]), 3),
            "facture": historique[j],
            "resume": _texte_facture(historique[j]),
        }
        for i, j in enumerate(ordre)
        if scores[j] > 0.15
    ]


def _similaires_tfidf(courante: dict, historique: list[dict], k: int) -> list[dict]:
    """Plan B pédagogique : même idée, vecteurs TF-IDF au lieu d'embeddings."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    docs = [_texte_facture(courante)] + [_texte_facture(f) for f in historique]
    X = TfidfVectorizer().fit_transform(docs)
    sims = cosine_similarity(X[0:1], X[1:]).ravel()
    ordre = np.argsort(sims)[::-1][:k]
    return [
        {
            "rang": i + 1,
            "score": round(float(sims[j]), 3),
            "facture": historique[j],
            "resume": _texte_facture(historique[j]),
            "methode": "tfidf",
        }
        for i, j in enumerate(ordre)
    ]


def carte_pca(factures: list[dict]) -> list[dict] | None:
    """
    Réduction de dimension : on passe de 768 dimensions → 2
    pour VOIR les paquets de factures. PCA = axes de plus grande variance.
    """
    if len(factures) < 3:
        return None
    M, textes = _matrice(factures)
    if M is None:
        return None
    from sklearn.decomposition import PCA

    xy = PCA(n_components=2, random_state=42).fit_transform(M)
    out = []
    for i, f in enumerate(factures):
        out.append(
            {
                "x": round(float(xy[i, 0]), 3),
                "y": round(float(xy[i, 1]), 3),
                "label": str(f.get("invoice_no") or f.get("seller_name") or i),
                "vendeur": f.get("seller_name") or "",
            }
        )
    return out


def vendeurs_proches(noms: list[str], seuil: float = 0.82) -> list[dict]:
    """Entity resolution : 'Dell Inc' ≈ 'DELL CORPORATION'."""
    uniques = [n for n in { _nettoyer(n) for n in noms if n } ]
    if len(uniques) < 2:
        return []
    vecs = [(n, embed(n)) for n in uniques]
    if any(v is None for _, v in vecs):
        return []
    paires = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            s = cosine(vecs[i][1], vecs[j][1])
            if s >= seuil:
                paires.append({"a": vecs[i][0], "b": vecs[j][0], "score": round(s, 3)})
    return sorted(paires, key=lambda p: -p["score"])


def repondre_rag(question: str, factures: list[dict], k: int = 4) -> dict:
    if not factures:
        return {"reponse": "Aucune facture en base.", "sources": []}

    faux = {"invoice_no": "QUERY", "seller_name": question, "client_name": question}
    hits = factures_similaires(faux, factures, k=k)
    sources = hits or [
        {"rang": i + 1, "score": None, "facture": f, "resume": _texte_facture(f)}
        for i, f in enumerate(factures[:k])
    ]
    contexte = json.dumps(
        [{"score": s.get("score"), "facture": s["facture"]} for s in sources],
        ensure_ascii=False,
        default=str,
    )[:10000]
    try:
        texte = appeler_llm_texte(
            "Tu es un analyste de factures. Réponds en français, concis. "
            "Utilise UNIQUEMENT le CONTEXTE. Si ce n'est pas dedans, dis-le. "
            "Cite les n° de facture utilisés.",
            f"CONTEXTE (factures les plus proches) :\n{contexte}\n\nQUESTION : {question}",
        )
    except ExtractionLLMError as e:
        texte = f"LLM indisponible : {e}"
    return {"reponse": texte, "sources": sources}