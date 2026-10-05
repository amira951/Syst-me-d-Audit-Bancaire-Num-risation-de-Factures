from src.preprocessing import nettoyer_image_depuis_chemin
from src.ocr import extraire_texte
from src.extraction import extraire_infos_facture, extraire_items
from src.table_extraction import (
    extraire_seller_client_par_position,
    parser_bloc_entite,
    extraire_items_par_position,
)

chemin = "data/raw/dataset_factures/batch_1/batch_1/batch1_1/batch1-0002.jpg"

# Seller / Client par position
resultat_position = extraire_seller_client_par_position(chemin)
seller_infos = parser_bloc_entite(resultat_position["seller_bloc"])
client_infos = parser_bloc_entite(resultat_position["client_bloc"])

# Texte global pour numéro, date, totaux, tax id, iban
image_nettoyee = nettoyer_image_depuis_chemin(chemin)
texte = extraire_texte(image_nettoyee)
infos, texte_corrige = extraire_infos_facture(texte)

infos["seller_name"] = seller_infos["nom"]
infos["seller_address"] = seller_infos["adresse"]
infos["client_name"] = client_infos["nom"]
infos["client_address"] = client_infos["adresse"]

# Items : position en priorité, regex en secours si la zone n'est pas trouvée
items = extraire_items_par_position(chemin)
if not items:
    items = extraire_items(texte_corrige)

print("----- INFOS FINALES -----")
print(infos)

print("\n----- ITEMS -----")
for item in items:
    print(item)