import re

CORRECTIONS_OCR = {
    "Setter:": "Seller:", "Sotter:": "Seller:",
    "Tox Id": "Tax Id", "Tox 1d": "Tax Id", "Tax 1d": "Tax Id",
    "Ket worth": "Net worth", "Not worth": "Net worth", "“et price": "Net price",
    "Crone werth": "Gross worth", "Cross werth": "Gross worth",
    "Gross werth": "Gross worth", "Groes worth": "Gross worth",
    "Date of Issue:": "Date of issue:",
    "Totes": "Total",
    " exch ": " each ",
    " eac ": " each ",
}


def corriger_texte(texte):
    for erreur, correction in CORRECTIONS_OCR.items():
        texte = texte.replace(erreur, correction)
    return texte


def extraire_infos_facture(texte):
    texte = corriger_texte(texte)
    infos = {}

    match = re.search(r"invoice\s*no:?\s*(\d+)", texte, re.IGNORECASE)
    infos["invoice_no"] = match.group(1) if match else None

    match = re.search(r"(\d{2}/\d{2}/\d{4})", texte)
    infos["date"] = match.group(1) if match else None

    tax_ids = re.findall(r"tax\s*id:?\s*([\d\-]+)", texte, re.IGNORECASE)
    infos["seller_tax_id"] = tax_ids[0] if len(tax_ids) > 0 else None
    infos["client_tax_id"] = tax_ids[1] if len(tax_ids) > 1 else None

    match = re.search(r"IBAN:?\s*([A-Z0-9]+)", texte, re.IGNORECASE)
    infos["iban"] = match.group(1) if match else None

    match = re.search(
        r"total\D*?\$?\s*([\d\s]+[.,]\d{2})\D*?\$?\s*([\d\s]+[.,]\d{2})\D*?\$?\s*([\d\s]+[.,]\d{2})",
        texte, re.IGNORECASE
    )
    if match:
        infos["net_worth_total"] = match.group(1).replace(" ", "")
        infos["vat_total"] = match.group(2).replace(" ", "")
        infos["gross_worth_total"] = match.group(3).replace(" ", "")
    else:
        infos["net_worth_total"] = None
        infos["vat_total"] = None
        infos["gross_worth_total"] = None

    return infos, texte


def extraire_items(texte_corrige):
    nombre = r"[\d]+(?:[\s.,]\d+)*"

    pattern = re.compile(
        rf"(\d+)\.\s+(.+?)\s+({nombre})\s+each\s+({nombre})\s+({nombre})\s+(\d+%)\s+({nombre})",
        re.DOTALL
    )
    items = []
    for match in pattern.finditer(texte_corrige):
        items.append({
            "numero": match.group(1),
            "description": re.sub(r"\s+", " ", match.group(2)).strip(),
            "qty": match.group(3).replace(" ", ""),
            "net_price": match.group(4).replace(" ", ""),
            "net_worth": match.group(5).replace(" ", ""),
            "vat_percent": match.group(6),
            "gross_worth": match.group(7).replace(" ", "")
        })
    return items