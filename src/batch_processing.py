import os
import glob
from src.preprocessing import nettoyer_image_depuis_chemin
from src.ocr import extraire_texte
from src.extraction import extraire_infos_facture
from src.database import initialiser_base, enregistrer_facture

def traiter_dossier(dossier_images):
    initialiser_base()
    chemins_images = glob.glob(os.path.join(dossier_images, "*.jpg"))
    print(f"{len(chemins_images)} image(s) trouvée(s) dans {dossier_images}")

    for chemin in chemins_images:
        print(f"Traitement de {os.path.basename(chemin)}...")
        try:
            image_nettoyee = nettoyer_image_depuis_chemin(chemin)
            texte = extraire_texte(image_nettoyee)
            infos = extraire_infos_facture(texte, chemin_image=chemin)
            enregistrer_facture(infos, chemin, texte)
            print(f"  -> Invoice no: {infos.get('invoice_no')} | Seller: {infos.get('seller')} | Client: {infos.get('client')}")
        except Exception as e:
            print(f"  Erreur sur {chemin} : {e}")

    print("Traitement terminé.")


if __name__ == "__main__":
    traiter_dossier("data/raw/dataset_factures/batch_1/batch_1/batch1_1")