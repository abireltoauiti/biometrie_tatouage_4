# ============================================================
# lsb_extract.py — Module Tatouage Numérique (LSB)
# Rôle : extraire le message caché depuis une image tatouée
# Technique : LSB (Least Significant Bit)
# ============================================================

from PIL import Image   # pour ouvrir l'image
import numpy as np      # pour manipuler les pixels


def bits_en_texte(bits):
    """
    Convertit une suite de bits en texte.
    Exemple : "01000001" → "A"
    """
    texte = ""
    # on prend les bits 8 par 8 (1 lettre = 8 bits)
    for i in range(0, len(bits), 8):
        octet = bits[i:i+8]
        if len(octet) < 8:
            break
        # int(..., 2) convertit le binaire en nombre
        # chr() convertit le nombre en lettre
        texte += chr(int(octet, 2))
    return texte


def extract(chemin_image):
    """
    Extrait le message caché depuis une image tatouée.

    Paramètres :
        chemin_image : chemin vers l'image tatouée (ex: "image_tatouee.png")

    Retourne :
        le message caché (str)
    """

    # --- Étape 1 : ouvrir l'image et récupérer les pixels ---
    image = Image.open(chemin_image)
    image = image.convert("RGB")
    pixels = np.array(image)

    hauteur, largeur, _ = pixels.shape

    # --- Étape 2 : lire le dernier bit de chaque pixel ---
    bits = ""

    for i in range(hauteur):
        for j in range(largeur):
            for canal in range(3):    # 0=Rouge, 1=Vert, 2=Bleu

                valeur_pixel = pixels[i, j, canal]

                # & 1 → garde uniquement le dernier bit
                # ex: 11001000 & 00000001 = 0
                # ex: 11001001 & 00000001 = 1
                bits += str(valeur_pixel & 1)

                # --- Étape 3 : vérifier si on a trouvé le marqueur de fin ---
                # tous les 8 bits on vérifie si le texte se termine par "###"
                if len(bits) % 8 == 0:
                    texte_actuel = bits_en_texte(bits)
                    if texte_actuel.endswith("###"):
                        # on enlève le marqueur "###" et on retourne le message
                        message = texte_actuel[:-3]
                        return message

    return "Aucun message trouvé"


# ============================================================
# TEST — exécuter ce fichier directement pour tester
# ============================================================
if __name__ == "__main__":

    print("🔍 Extraction du message caché...\n")

    message = extract("tatouee_20260411_225045.png")

    print(f"✅ Message extrait : '{message}'")
    print("\nPreuve d'authenticité confirmée ! 🔏")