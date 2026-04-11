# ============================================================
# capture.py — Capture d'image via webcam
# Rôle : capturer une image et l'envoyer directement à lsb_embed
# ============================================================

import cv2                  # pour la webcam
from datetime import datetime
from lsb_embed import embed  # on importe directement notre fonction


def capturer_et_tatouer(id_camera="CAM_01", statut="INCONNU", nom="INCONNU"):
    """
    Ouvre la webcam, affiche le flux en direct,
    capture une photo avec ESPACE et la tatouée automatiquement.

    Paramètres :
        id_camera : identifiant de la caméra (ex: "CAM_01")
        statut    : statut de la personne (ex: "AUTORISÉ", "INTERDIT", "INCONNU")
        nom       : nom de la personne si connue (ex: "Ahmed")
    """

    # --- Étape 1 : ouvrir la webcam ---
    cap = cv2.VideoCapture(0)   # 0 = webcam par défaut de ton PC

    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam !")
        return None

    print("📷 Webcam ouverte !")
    print("   → Appuie sur ESPACE pour capturer")
    print("   → Appuie sur Q pour quitter\n")

    image_capturee = None

    # --- Étape 2 : afficher le flux en direct ---
    while True:
        ret, frame = cap.read()   # lit une frame de la webcam

        if not ret:
            print("❌ Erreur lecture webcam")
            break

        # affiche le flux vidéo dans une fenêtre
        cv2.imshow("Webcam — ESPACE pour capturer | Q pour quitter", frame)

        # --- Étape 3 : attendre une touche ---
        touche = cv2.waitKey(1) & 0xFF

        if touche == ord(' '):    # ESPACE → capturer
            image_capturee = frame
            print("✅ Image capturée !")
            break

        elif touche == ord('q'):  # Q → quitter sans capturer
            print("❌ Capture annulée")
            break

    # --- Étape 4 : fermer la webcam ---
    cap.release()
    cv2.destroyAllWindows()

    if image_capturee is None:
        return None

    # --- Étape 5 : sauvegarder l'image capturée ---
    chemin_capture = "capture_brute.png"
    cv2.imwrite(chemin_capture, image_capturee)
    print(f"📁 Image sauvegardée : {chemin_capture}")

    # --- Étape 6 : construire le message de tatouage ---
    maintenant = datetime.now()
    message = (
        f"{id_camera} | "
        f"{maintenant.strftime('%Y-%m-%d')} | "
        f"{maintenant.strftime('%H:%M:%S')} | "
        f"{nom} | "
        f"{statut}"
    )

    print(f"\n🔏 Tatouage en cours...")
    print(f"   Message : {message}")

    # --- Étape 7 : tatouer directement l'image capturée ---
    chemin_sortie = f"tatouee_{maintenant.strftime('%Y%m%d_%H%M%S')}.png"

    embed(
        chemin_image=chemin_capture,
        message=message,
        chemin_sortie=chemin_sortie
    )

    print(f"\n🎯 Pipeline complet :")
    print(f"   Webcam → capture_brute.png → {chemin_sortie}")
    print(f"   Message caché : '{message}'")

    return chemin_sortie


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":

    chemin = capturer_et_tatouer(
        id_camera="CAM_01",
        statut="INCONNU",     # change selon le cas
        nom="INCONNU"         # change selon le cas
    )

    if chemin:
        print(f"\n✅ Image tatouée prête : {chemin}")
        print("   Lance lsb_extract.py pour vérifier !")