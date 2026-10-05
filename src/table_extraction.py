import pytesseract
import pandas as pd
import cv2
from src.preprocessing import charger_image, extraire_zone, nettoyer_image
from src.ocr import extraire_texte
import re


def extraire_donnees_positionnees(image_nettoyee, langue="eng"):
    """
    Au lieu de renvoyer un texte brut, renvoie chaque mot détecté
    avec sa position (ligne, colonne approximative) sur l'image.
    """
    donnees = pytesseract.image_to_data(
        image_nettoyee, lang=langue, output_type=pytesseract.Output.DATAFRAME
    )
    donnees = donnees[donnees.text.notna()]
    donnees = donnees[donnees.text.str.strip() != ""]
    return donnees


def regrouper_par_ligne(donnees, tolerance_y=10):
    """
    Regroupe les mots à peu près à la même hauteur en une ligne reconstituée.
    """
    donnees = donnees.sort_values(by=["top", "left"])
    lignes = []
    ligne_courante = []
    y_reference = None

    for _, mot in donnees.iterrows():
        if y_reference is None or abs(mot["top"] - y_reference) <= tolerance_y:
            ligne_courante.append(mot)
            y_reference = mot["top"] if y_reference is None else y_reference
        else:
            lignes.append(ligne_courante)
            ligne_courante = [mot]
            y_reference = mot["top"]

    if ligne_courante:
        lignes.append(ligne_courante)

    lignes_texte = []
    for ligne in lignes:
        ligne_triee = sorted(ligne, key=lambda m: m["left"])
        texte_ligne = " ".join(m["text"] for m in ligne_triee)
        lignes_texte.append(texte_ligne)

    return lignes_texte


def extraire_seller_client_par_position(chemin_image):
    """
    Découpe l'image en colonne gauche (Seller) et colonne droite (Client)
    AVANT l'OCR, pour éviter toute fusion de texte.
    """
    image = charger_image(chemin_image)

    zone_gauche = extraire_zone(image, 0.0, 0.15, 0.5, 0.35)
    zone_droite = extraire_zone(image, 0.5, 0.15, 1.0, 0.35)

    texte_gauche = extraire_texte(nettoyer_image(zone_gauche))
    texte_droite = extraire_texte(nettoyer_image(zone_droite))

    return {
        "seller_bloc": texte_gauche.strip(),
        "client_bloc": texte_droite.strip()
    }


def parser_bloc_entite(bloc_texte):
    """
    Extrait nom, adresse et infos d'un bloc Seller ou Client déjà isolé
    (plus besoin de LLM puisque le texte n'est plus fusionné).
    """
    lignes = [l.strip() for l in bloc_texte.split("\n") if l.strip()]

    lignes = [l for l in lignes if not re.fullmatch(r"[^a-zA-Z]*(Seller|Client)\s*:?", l, re.IGNORECASE)]

    nom = lignes[0] if lignes else None

    adresse_lignes = []
    for ligne in lignes[1:]:
        if re.search(r"tax\s*id", ligne, re.IGNORECASE):
            break
        adresse_lignes.append(ligne)
    adresse = ", ".join(adresse_lignes) if adresse_lignes else None

    return {"nom": nom, "adresse": adresse}


def extraire_items_par_position(chemin_image):
    """
    Reconstruit le tableau ITEMS. Au lieu de dépendre du mot "each"
    (peu fiable en OCR sur cette qualité d'image), on identifie les lignes
    de valeurs par la présence d'AU MOINS 2 montants numériques, puis on
    reconstruit les valeurs manquantes par cohérence comptable :
    Net worth = Qty × Net price, Gross worth = Net worth × (1 + VAT%).
    Les lignes suivantes sans montants sont recollées à la description
    de l'article précédent (descriptions multi-lignes).
    """
    from src.preprocessing import nettoyer_image_depuis_chemin

    image_nettoyee = nettoyer_image_depuis_chemin(chemin_image)
    donnees = extraire_donnees_positionnees(image_nettoyee)

    mot_items = donnees[donnees["text"].str.upper() == "ITEMS"]
    mot_summary = donnees[donnees["text"].str.upper() == "SUMMARY"]

    if mot_items.empty or mot_summary.empty:
        return []

    y_debut = int(mot_items["top"].min())
    y_fin = int(mot_summary["top"].min())

    hauteur, largeur = image_nettoyee.shape[:2]
    y1 = max(0, y_debut - 15)
    y2 = min(hauteur, y_fin + 15)
    zone_image = image_nettoyee[y1:y2, 0:largeur]

    zone_zoomee = cv2.resize(zone_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, zone_zoomee = cv2.threshold(zone_zoomee, 127, 255, cv2.THRESH_BINARY)

    donnees_zone = extraire_donnees_positionnees(zone_zoomee)
    lignes_texte = regrouper_par_ligne(donnees_zone, tolerance_y=20)

    print("----- LIGNES RECONSTRUITES (cohérence comptable) -----")
    for l in lignes_texte:
        print(repr(l))
    print("----- FIN DEBUG -----")

    nombre = r"\d{1,3}(?:[\s.,]\d{3})*[.,]\d{2}"

    def to_float(v):
        """Convertit '1 400,00' ou '139.95' en float, quel que soit le séparateur."""
        v = v.replace(" ", "")
        if "," in v or "." in v:
            pos = max(v.rfind(","), v.rfind("."))
            entier = re.sub(r"[.,]", "", v[:pos])
            decimal = v[pos + 1:]
            return float(f"{entier}.{decimal}")
        return float(v)

    def format_val(f):
        """Reformate un float en chaîne avec virgule décimale, style facture."""
        return f"{f:.2f}".replace(".", ",")

    items = []
    numero = 0
    dernier_item = None

    for ligne in lignes_texte:
        nombres_trouves = re.findall(nombre, ligne)

        if len(nombres_trouves) < 2:
            # Pas assez de nombres : soit du bruit (en-tête, séparateur),
            # soit une suite de description de l'article précédent.
            ligne_propre = ligne.strip()
            if dernier_item and len(ligne_propre) > 3 and not re.match(r"^[\W_]+$", ligne_propre):
                dernier_item["description"] += " " + ligne_propre
            continue

        numero += 1

        vat_match = re.search(r"(\d+)\s*%", ligne)
        vat_percent_num = int(vat_match.group(1)) if vat_match else 10

        index_coupure = ligne.find(nombres_trouves[0])
        description = ligne[:index_coupure].strip()
        description = re.sub(r"^\d+[.,]?\s*", "", description).strip()

        valeurs = [re.match(r"[\d.,]+", n.replace(" ", "")).group() for n in nombres_trouves]

        qty = net_price = net_worth = gross_worth = None

        try:
            if len(valeurs) >= 4:
                qty, net_price, net_worth, gross_worth = valeurs[0], valeurs[1], valeurs[2], valeurs[3]
            elif len(valeurs) == 3:
                net_price, net_worth, gross_worth = valeurs
                qty = format_val(round(to_float(net_worth) / to_float(net_price)))
            elif len(valeurs) == 2:
                net_price, net_worth = valeurs
                gross_worth = format_val(to_float(net_worth) * (1 + vat_percent_num / 100))

            net_price = format_val(to_float(net_price)) if net_price else None
            net_worth = format_val(to_float(net_worth)) if net_worth else None
            gross_worth = format_val(to_float(gross_worth)) if gross_worth else None
        except (ValueError, ZeroDivisionError, AttributeError):
            pass

        dernier_item = {
            "numero": str(numero),
            "description": description,
            "qty": qty,
            "net_price": net_price,
            "net_worth": net_worth,
            "vat_percent": f"{vat_percent_num}%",
            "gross_worth": gross_worth,
        }
        items.append(dernier_item)

    return items