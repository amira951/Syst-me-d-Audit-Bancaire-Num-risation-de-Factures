"""
Toute IA doit donner une INCERTITUDE, pas seulement une réponse.
Tesseract fournit déjà une confiance 0-100 par mot : on l'exploite.

Fusion de scores (ensemble / décision) :
    score = w1·OCR + w2·règles métier + w3·IBAN + w4·source
C'est le même principe qu'un comité de classifieurs.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytesseract


def analyser_confiance_ocr(chemin_image: str, langue: str = "eng", seuil: int = 60) -> dict:
    img = cv2.imread(chemin_image)
    if img is None:
        raise FileNotFoundError(chemin_image)
    annot = img.copy()

    df = pytesseract.image_to_data(img, lang=langue, output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df[df["conf"] >= 0]
    if df.empty:
        return {"moyenne": 0.0, "nb_mots": 0, "nb_faibles": 0, "mots_faibles": [], "png": None}

    faibles = []
    for _, row in df.iterrows():
        conf = float(row["conf"])
        x, y, w, h = int(row["left"]), int(row["top"]), int(row["width"]), int(row["height"])
        if conf >= 80:
            color = (46, 160, 67)      # vert
        elif conf >= seuil:
            color = (0, 165, 255)      # orange
        else:
            color = (50, 50, 220)      # rouge
            faibles.append({"texte": str(row["text"]), "conf": round(conf, 1)})
        cv2.rectangle(annot, (x, y), (x + w, y + h), color, 2)

    ok, buf = cv2.imencode(".png", annot)
    return {
        "moyenne": round(float(df["conf"].mean()), 1),
        "mediane": round(float(df["conf"].median()), 1),
        "nb_mots": int(len(df)),
        "nb_faibles": len(faibles),
        "mots_faibles": sorted(faibles, key=lambda m: m["conf"])[:12],
        "png": buf.tobytes() if ok else None,
    }


def valider_iban(iban: str | None) -> dict:
    """ISO 13616 : déplacer 4 1ers caractères, A=10…Z=35, puis mod 97 == 1."""
    import re

    brut = re.sub(r"\s+", "", iban or "").upper()
    if not brut:
        return {"present": False, "valide": None, "iban": None}
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", brut):
        return {"present": True, "valide": False, "iban": brut, "raison": "format"}
    rearr = brut[4:] + brut[:4]
    chiffres = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearr)
    ok = int(chiffres) % 97 == 1
    return {"present": True, "valide": ok, "iban": brut, "raison": None if ok else "checksum mod 97"}


def score_global(
    ocr_moyenne: float,
    pct_lignes_ok: float,
    iban: dict,
    source_tableau: str,
) -> dict:
    s_ocr = max(0.0, min(ocr_moyenne / 100.0, 1.0))
    s_metier = max(0.0, min(pct_lignes_ok, 1.0))
    if not iban.get("present"):
        s_iban = 0.7
    else:
        s_iban = 1.0 if iban.get("valide") else 0.15
    s_src = {"llm": 1.0, "position": 0.7, "regex": 0.45}.get(source_tableau, 0.5)

    score = 100 * (0.35 * s_ocr + 0.35 * s_metier + 0.15 * s_iban + 0.15 * s_src)
    if score >= 80:
        niveau, couleur = "élevée", "green"
    elif score >= 60:
        niveau, couleur = "moyenne", "orange"
    else:
        niveau, couleur = "faible — revue humaine", "red"
    return {
        "score": round(score, 1),
        "niveau": niveau,
        "couleur": couleur,
        "detail": {
            "ocr": round(100 * s_ocr, 1),
            "regles_metier": round(100 * s_metier, 1),
            "iban": round(100 * s_iban, 1),
            "source_tableau": round(100 * s_src, 1),
        },
    }