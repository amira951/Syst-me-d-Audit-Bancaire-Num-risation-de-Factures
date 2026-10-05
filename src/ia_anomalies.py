"""
Apprentissage NON SUPERVISÉ.

Isolation Forest : on isole les points 'rares' en coupant l'espace
au hasard. Un point anormal est isolé en PEU de coupes.
Idéal en banque : on n'a presque jamais d'exemples de fraude labellisés.

Deux niveaux :
  1. Intra-facture  : une ligne beaucoup plus chère que les autres (z-score)
  2. Inter-factures : une facture bizarre par rapport à l'historique
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def _f(x):
    try:
        return float(str(x).replace(" ", "").replace(",", ".").replace("€", "").replace("%", ""))
    except (TypeError, ValueError):
        return None


def anomalies_lignes(items: list[dict], z_seuil: float = 2.2) -> list[dict]:
    """Z-score : (x - moyenne) / écart-type. Base de la stat en IA."""
    montants = [_f(i.get("gross_worth") or i.get("net_worth")) for i in items]
    valides = [m for m in montants if m is not None]
    if len(valides) < 3:
        return []

    mu = float(np.mean(valides))
    sigma = float(np.std(valides)) or 1.0
    out = []
    for it, m in zip(items, montants):
        if m is None:
            continue
        z = (m - mu) / sigma
        if abs(z) >= z_seuil:
            out.append(
                {
                    "numero": it.get("numero"),
                    "description": it.get("description"),
                    "montant": m,
                    "z_score": round(z, 2),
                    "raison": f"montant atypique (z={z:.2f}, moyenne={mu:.2f})",
                }
            )
    return out


def _vecteur_facture(infos: dict, items: list[dict] | None = None) -> list[float]:
    items = items or []
    ttc = _f(infos.get("gross_worth_total")) or 0.0
    ht = _f(infos.get("net_worth_total")) or 0.0
    tva = _f(infos.get("vat_total")) or 0.0
    n = len(items) or 1
    max_ligne = 0.0
    for it in items:
        max_ligne = max(max_ligne, _f(it.get("gross_worth")) or 0.0)
    return [ttc, ht, tva, float(n), max_ligne, (tva / ht) if ht else 0.0]


def anomalies_historique(
    historique: list[dict],
    facture_courante: dict,
    items_courants: list[dict] | None = None,
) -> dict:
    """
    Isolation Forest sur l'historique.
    contamination='auto' : le modèle estime lui-même la proportion d'outliers.
    """
    if len(historique) < 8:
        return {
            "disponible": False,
            "message": f"Il faut ~8 factures en historique (actuellement {len(historique)}). "
            "C'est normal : un modèle non supervisé a besoin d'une distribution.",
        }

    X_hist = np.array([_vecteur_facture(f) for f in historique], dtype=float)
    x0 = np.array([_vecteur_facture(facture_courante, items_courants)], dtype=float)

    clf = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    clf.fit(X_hist)
    pred = int(clf.predict(x0)[0])          # -1 = anomalie, 1 = normal
    score = float(clf.decision_function(x0)[0])  # plus négatif = plus bizarre

    return {
        "disponible": True,
        "est_anomalie": pred == -1,
        "score": round(score, 3),
        "interpretation": (
            "Facture atypique par rapport à l'historique (montant, TVA ou nb d'articles)."
            if pred == -1
            else "Facture dans le profil habituel de l'historique."
        ),
    }