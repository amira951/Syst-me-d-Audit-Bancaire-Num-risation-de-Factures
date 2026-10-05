import cv2
import pytesseract

# 1. Ouvrir l'image (vraie facture du dataset)
image = cv2.imread("dataset_factures/batch_1/batch_1/batch1_1/batch1-0001.jpg")

print(f"Largeur : {image.shape[1]} pixels")
print(f"Hauteur : {image.shape[0]} pixels")

# 2. Convertir en niveaux de gris
image_grise = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("facture_grise.png", image_grise)

# 3. Binariser
seuil, image_binaire = cv2.threshold(image_grise, 128, 255, cv2.THRESH_BINARY)
cv2.imwrite("facture_binaire.png", image_binaire)

# 4. OCR
texte = pytesseract.image_to_string(image_binaire, lang="eng")

print("----- TEXTE EXTRAIT -----")
print(texte)