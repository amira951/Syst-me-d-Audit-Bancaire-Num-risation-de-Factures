import pytesseract

def extraire_texte(image_nettoyee, langue="eng", psm=6):
    """
    Extrait le texte d'une image nettoyée.
    psm=6 : bloc de texte uniforme (bon défaut pour factures et colonnes isolées).
    """
    config = f"--psm {psm}"
    return pytesseract.image_to_string(image_nettoyee, lang=langue, config=config)

