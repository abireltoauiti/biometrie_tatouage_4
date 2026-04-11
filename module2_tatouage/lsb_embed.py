# ============================================================
# lsb_embed.py — Module Tatouage Numérique (LSB)
# Rôle : cacher un message texte dans une image
# Technique : LSB (Least Significant Bit)
# ============================================================

from PIL import Image  
import numpy as np     

def texte_en_bits(texte):
    """
    Convertit un texte en une suite de bits (0 et 1).
    Exemple : "A" → "01000001"
    """
    bits = ""
    for lettre in texte:
        # ord() donne le code numérique de la lettre (ex: 'A' → 65)
        # format(..., '08b') convertit en binaire sur 8 bits
        bits += format(ord(lettre), '08b')
    return bits


def embed(chemin_image, message, chemin_sortie):
    """
    Cache un message dans une image via LSB.

    Paramètres :
        chemin_image  : chemin vers l'image originale (ex: "test_image.png")
        message       : le texte à cacher (ex: "CAM_01|2025-04-11|14:32")
        chemin_sortie : où sauvegarder l'image tatouée (ex: "image_tatouee.png")
    """

    # --- Étape 1 : ouvrir l'image et la convertir en tableau de pixels ---
    image = Image.open(chemin_image)
    image = image.convert("RGB")           # s'assurer qu'on est en mode RGB
    pixels = np.array(image)               # tableau numpy de forme (hauteur, largeur, 3)

    # --- Étape 2 : préparer le message ---
    # On ajoute un marqueur de fin "###" pour savoir où s'arrête le message
    message_complet = message + "###"
    bits = texte_en_bits(message_complet)
    nb_bits = len(bits)

    # --- Étape 3 : vérifier que l'image est assez grande ---
    hauteur, largeur, _ = pixels.shape
    capacite = hauteur * largeur * 3       # 3 canaux RGB = 3 bits par pixel
    if nb_bits > capacite:
        raise ValueError(f"Message trop long ! Capacité max : {capacite // 8} caractères")

    print(f"Message à cacher : '{message}'")
    print(f"Nombre de bits   : {nb_bits}")
    print(f"Capacité image   : {capacite} bits")

    # --- Étape 4 : cacher les bits dans les pixels ---
    index_bit = 0   # on commence au premier bit du message

    for i in range(hauteur):
        for j in range(largeur):
            for canal in range(3):          # 0=Rouge, 1=Vert, 2=Bleu

                if index_bit >= nb_bits:    # tous les bits sont cachés → on arrête
                    break

                valeur_pixel = pixels[i, j, canal]   # ex: 201 → 11001001

                # On remplace le dernier bit par le bit du message :
                # & 254  →  met le dernier bit à 0  (254 = 11111110)
                # | bit  →  met le dernier bit à la valeur voulue (0 ou 1)
                bit = int(bits[index_bit])
                pixels[i, j, canal] = (valeur_pixel & 254) | bit

                index_bit += 1

            if index_bit >= nb_bits:
                break
        if index_bit >= nb_bits:
            break

    # --- Étape 5 : sauvegarder l'image tatouée ---
    image_tatouee = Image.fromarray(pixels)
    image_tatouee.save(chemin_sortie)

    print(f"\n✅ Image tatouée sauvegardée : {chemin_sortie}")
    print(f"   Visuellement identique à l'originale !")


# ============================================================
# TEST — exécuter ce fichier directement pour tester
# ============================================================
if __name__ == "__main__":

    from datetime import datetime

    # Message à cacher (comme dans le scénario de surveillance)
    message = (
        f"CAM_01 | "
        f"{datetime.now().strftime('%Y-%m-%d')} | "
        f"{datetime.now().strftime('%H:%M:%S')} | "
        f"INTRUS"
    )
    embed(
        chemin_image="test_1.jpg",
        message=message,
        chemin_sortie="image_tatouee1.png"
    )