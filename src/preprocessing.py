import cv2

def charger_image(chemin_image):
    """Charge une image depuis le disque."""
    image = cv2.imread(chemin_image)
    if image is None:
        raise ValueError(f"Impossible de lire l'image : {chemin_image}")
    return image


def nettoyer_image(image):
    """
    Nettoie une image DÉJÀ CHARGÉE (tableau numpy) pour préparer l'OCR :
    agrandissement, réduction de bruit, contraste local, binarisation adaptative.
    """
    if len(image.shape) == 3:
        image_grise = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_grise = image

    image_grise = cv2.resize(
        image_grise, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
    )
    image_debruitee = cv2.fastNlMeansDenoising(image_grise, h=10)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_contrastee = clahe.apply(image_debruitee)

    image_binaire = cv2.adaptiveThreshold(
        image_contrastee,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=35,
        C=15
    )
    return image_binaire


def nettoyer_image_depuis_chemin(chemin_image):
    """Fonction de confort : charge + nettoie en un seul appel."""
    image = charger_image(chemin_image)
    return nettoyer_image(image)


def extraire_zone(image, x1_frac, y1_frac, x2_frac, y2_frac):
    """
    Découpe une zone rectangulaire de l'image, en fractions (0.0 à 1.0)
    de la largeur/hauteur totale. Permet d'isoler par exemple la colonne
    Seller (gauche) ou Client (droite) avant l'OCR.
    """
    hauteur, largeur = image.shape[:2]
    x1, x2 = int(largeur * x1_frac), int(largeur * x2_frac)
    y1, y2 = int(hauteur * y1_frac), int(hauteur * y2_frac)
    return image[y1:y2, x1:x2]