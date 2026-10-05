import json
import re
from typing import Any

import ollama

MODELES_CANDIDATS = [
    "qwen2.5:7b",
    "qwen2.5",
    "llama3.1",
    "mistral-nemo",
    "mistral",
]

CLES_ITEM = [
    "numero",
    "description",
    "qty",
    "um",
    "net_price",
    "net_worth",
    "vat_percent",
    "gross_worth",
]


class ExtractionLLMError(Exception):
    pass


def _noms_modeles_installes() -> list[str]:
    try:
        res = ollama.list()
        modeles = getattr(res, "models", None)
        if modeles is None and isinstance(res, dict):
            modeles = res.get("models", [])
        noms = []
        for m in modeles or []:
            nom = getattr(m, "model", None)
            if nom is None and isinstance(m, dict):
                nom = m.get("model") or m.get("name")
            if nom:
                noms.append(nom)
        return noms
    except Exception:
        return []


def choisir_modele() -> str | None:
    installes = _noms_modeles_installes()
    if not installes:
        return None
    for candidat in MODELES_CANDIDATS:
        for installe in installes:
            if installe == candidat or installe.startswith(candidat + ":") or candidat in installe:
                return installe
    return installes[0]


def statut_ollama() -> dict:
    modele = choisir_modele()
    return {
        "disponible": modele is not None,
        "modele": modele,
        "installes": _noms_modeles_installes(),
    }


def _contenu_reponse(reponse: Any) -> str:
    msg = reponse["message"] if isinstance(reponse, dict) else reponse.message
    return (msg["content"] if isinstance(msg, dict) else msg.content).strip()


def _nettoyer_json(texte: str) -> Any:
    if not texte:
        raise ExtractionLLMError("Réponse LLM vide")

    texte = texte.strip()
    texte = re.sub(r"^```(?:json)?\s*", "", texte, flags=re.IGNORECASE)
    texte = re.sub(r"\s*```$", "", texte)

    try:
        return json.loads(texte)
    except json.JSONDecodeError:
        pass

    candidats = re.findall(r"(\{.*\}|\[.*\])", texte, flags=re.DOTALL)
    candidats.sort(key=len, reverse=True)
    for bloc in candidats:
        try:
            return json.loads(bloc)
        except json.JSONDecodeError:
            continue
    raise ExtractionLLMError(f"JSON illisible : {texte[:300]}")


def appeler_llm_json(prompt_systeme: str, prompt_user: str, modele: str | None = None) -> Any:
    modele = modele or choisir_modele()
    if not modele:
        raise ExtractionLLMError(
            "Aucun modèle Ollama. Lance : ollama serve  puis  ollama pull qwen2.5:7b"
        )

    try:
        reponse = ollama.chat(
            model=modele,
            format="json",
            options={"temperature": 0, "num_ctx": 8192},
            messages=[
                {"role": "system", "content": prompt_systeme},
                {"role": "user", "content": prompt_user},
            ],
        )
        return _nettoyer_json(_contenu_reponse(reponse))
    except ExtractionLLMError:
        raise
    except Exception as e:
        try:
            reponse = ollama.chat(
                model=modele,
                options={"temperature": 0, "num_ctx": 8192},
                messages=[
                    {"role": "system", "content": prompt_systeme},
                    {"role": "user", "content": prompt_user},
                ],
            )
            return _nettoyer_json(_contenu_reponse(reponse))
        except Exception:
            raise ExtractionLLMError(f"Ollama a échoué ({modele}) : {e}") from e


def appeler_llm_texte(prompt_systeme: str, prompt_user: str, modele: str | None = None) -> str:
    modele = modele or choisir_modele()
    if not modele:
        raise ExtractionLLMError("Aucun modèle Ollama disponible")
    reponse = ollama.chat(
        model=modele,
        options={"temperature": 0.2, "num_ctx": 8192},
        messages=[
            {"role": "system", "content": prompt_systeme},
            {"role": "user", "content": prompt_user},
        ],
    )
    return _contenu_reponse(reponse)


def extraire_bloc_items(texte_complet: str) -> str:
    m = re.search(
        r"(ITEMS|No\.?\s+Description|N°\s+Description).+",
        texte_complet,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return m.group(0).strip() if m else texte_complet


def _to_float(x):
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = (
        str(x)
        .strip()
        .replace("%", "")
        .replace("€", "")
        .replace("$", "")
        .replace("\u00a0", "")
        .replace(" ", "")
        .replace(",", ".")
    )
    try:
        return float(s)
    except ValueError:
        return None


def _fmt(val, decimales=2):
    f = _to_float(val)
    if f is None:
        return None if val in (None, "") else str(val)
    return f"{f:.{decimales}f}".replace(".", ",")


def normaliser_item(brut: dict) -> dict:
    alias = {
        "no": "numero",
        "n": "numero",
        "n°": "numero",
        "num": "numero",
        "quantite": "qty",
        "quantity": "qty",
        "unit_price": "net_price",
        "prix_unitaire": "net_price",
        "montant_ht": "net_worth",
        "vat": "vat_percent",
        "tva": "vat_percent",
        "montant_ttc": "gross_worth",
    }
    src = {alias.get(str(k).lower(), str(k).lower()): v for k, v in brut.items()}
    numero = src.get("numero")
    try:
        numero = int(float(numero)) if numero not in (None, "") else None
    except (TypeError, ValueError):
        pass

    return {
        "numero": numero,
        "description": " ".join((src.get("description") or "").split()) or None,
        "qty": _fmt(src.get("qty")),
        "um": src.get("um") or "each",
        "net_price": _fmt(src.get("net_price")),
        "net_worth": _fmt(src.get("net_worth")),
        "vat_percent": _fmt(src.get("vat_percent")),
        "gross_worth": _fmt(src.get("gross_worth")),
    }


PROMPT_LIGNES = """Tu es un extracteur de lignes de facture.
À partir du texte OCR de la section ITEMS, extraire TOUTES les lignes d'articles.

Règles strictes :
- La description d'un article s'étale souvent sur PLUSIEURS lignes : reconstitue-la sur une seule ligne.
- Ne fusionne JAMAIS deux articles. S'il y a 5 articles, renvoie exactement 5 objets.
- N'invente aucun article ni aucun montant.
- Les nombres européens utilisent la virgule : 5,00 et 1 600,00
- Convertis tous les montants en nombres JSON (point décimal)
- "1 600,00" → 1600.00   "10%" → 10.0   "5,00" → 5.0
- Si un champ est illisible : null
- Ignore l'en-tête et le bloc SUMMARY / Total

Réponds UNIQUEMENT avec ce JSON :
{
  "items": [
    {
      "numero": 1,
      "description": "nom du produit",
      "qty": 5.0,
      "um": "each",
      "net_price": 169.95,
      "net_worth": 849.75,
      "vat_percent": 10.0,
      "gross_worth": 934.73
    }
  ]
}
"""


def extraire_lignes_facture(texte_complet: str) -> list[dict]:
    bloc = extraire_bloc_items(texte_complet)
    data = appeler_llm_json(PROMPT_LIGNES, bloc)
    brut = data.get("items") if isinstance(data, dict) else data
    if not isinstance(brut, list):
        raise ExtractionLLMError("Le LLM n'a pas renvoyé une liste d'articles")
    return [normaliser_item(x) for x in brut if isinstance(x, dict)]


def valider_lignes(items: list[dict], tol: float = 0.1) -> list[dict]:
    """
    Auto-correction comptable :
    une facture obéit à qty×prix = HT  et  HT×(1+TVA) = TTC.
    On recalcule les valeurs douteuses à partir des valeurs fiables.
    """
    for it in items:
        q = _to_float(it.get("qty"))
        p = _to_float(it.get("net_price"))
        ht = _to_float(it.get("net_worth"))
        tva = _to_float(it.get("vat_percent"))
        ttc = _to_float(it.get("gross_worth"))

        corrections = []

        # 1) HT = qty × prix
        if q is not None and p is not None:
            ht_calc = q * p
            if ht is None or abs(ht_calc - ht) > max(tol, 0.02 * ht_calc):
                if ht is not None:
                    corrections.append(f"HT {ht:.2f}→{ht_calc:.2f}")
                ht = ht_calc

        # 2) Si TTC + TVA connus et HT incohérent → HT = TTC / (1+TVA)
        if ttc is not None and tva is not None:
            ht_ttc = ttc / (1 + tva / 100.0)
            if ht is None or abs(ht - ht_ttc) > max(1.0, 0.05 * ht_ttc):
                if ht is not None:
                    corrections.append(f"HT via TTC {ht:.2f}→{ht_ttc:.2f}")
                ht = ht_ttc
                if q not in (None, 0):
                    p_calc = ht / q
                    if p is None or abs(p - p_calc) > max(tol, 0.02 * p_calc):
                        if p is not None:
                            corrections.append(f"Prix {p:.2f}→{p_calc:.2f}")
                        p = p_calc

        # 3) TTC final propre
        if ht is not None and tva is not None:
            ttc_calc = ht * (1 + tva / 100.0)
            if ttc is None or abs(ttc - ttc_calc) > max(tol, 0.02 * ttc_calc):
                if ttc is not None:
                    corrections.append(f"TTC {ttc:.2f}→{ttc_calc:.2f}")
                ttc = ttc_calc

        # Réécriture des valeurs corrigées
        if q is not None:
            it["qty"] = _fmt(q)
        if p is not None:
            it["net_price"] = _fmt(p)
        if ht is not None:
            it["net_worth"] = _fmt(ht)
        if ttc is not None:
            it["gross_worth"] = _fmt(ttc)

        # Vérif finale
        coherent = True
        if q is not None and p is not None and ht is not None:
            if abs(q * p - ht) > max(tol, 0.02 * abs(ht)):
                coherent = False

        it["valide"] = coherent
        it["alertes"] = " ; ".join(corrections) if corrections else ""
        it["statut"] = "✅" if coherent else "⚠️"

    return items


PROMPT_CORRECTION = """Tu corriges le JSON d'articles d'une facture selon l'instruction humaine.
Ne change que ce qui est demandé. Conserve le même schéma.
Réponds UNIQUEMENT avec :
{ "items": [ { "numero": 1, "description": "", "qty": 0, "um": "each", "net_price": 0, "net_worth": 0, "vat_percent": 0, "gross_worth": 0 } ] }
Les montants sont des nombres JSON (point décimal).
"""


def corriger_lignes(items: list[dict], instruction: str) -> list[dict]:
    payload = [{k: it.get(k) for k in CLES_ITEM} for it in items]
    data = appeler_llm_json(
        PROMPT_CORRECTION,
        f"JSON actuel :\n{json.dumps(payload, ensure_ascii=False)}\n\nInstruction :\n{instruction}",
    )
    brut = data.get("items") if isinstance(data, dict) else data
    if not isinstance(brut, list):
        raise ExtractionLLMError("Correction LLM invalide")
    return valider_lignes([normaliser_item(x) for x in brut if isinstance(x, dict)])