#!/usr/bin/env python3
"""
Reconnaissance faciale + détection des émotions.
Version améliorée : netteté plus stricte + identification avec marge de similarité.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("OpenCV manque. Installez-le avec : pip install -r requirements.txt")

__version__ = "1.1.0"

# Seuils améliorés pour une meilleure identification
SEUIL_RECONNAISSANCE = 0.40
SEUIL_MARGE = 0.08
TAILLE_MIN_VISAGE = 80
TAILLE_MIN_IDENTIF = 50
NETTETE_MIN = 60.0
NETTETE_MIN_IDENTIF = 45.0

def nettete(image) -> float:
    """Netteté améliorée (flou gaussien + variance du Laplacien)."""
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())

def identifier(empreinte, connus, seuil=SEUIL_RECONNAISSANCE, marge=SEUIL_MARGE):
    """Identification avec marge pour éviter les confusions entre personnes."""
    scores = []
    for nom, emps in connus.items():
        if len(emps) == 0:
            continue
        s = float(np.max(emps @ empreinte))
        scores.append((s, nom))
    if not scores:
        return "Inconnu", -1.0
    scores.sort(reverse=True)
    best_score, best = scores[0]
    if best_score < seuil:
        return "Inconnu", best_score
    if len(scores) > 1 and (best_score - scores[1][0]) < marge:
        return "Inconnu", best_score
    return best, best_score

if __name__ == "__main__":
    print("visage.py amélioré chargé")
    print(f"Seuil : {SEUIL_RECONNAISSANCE} | Marge : {SEUIL_MARGE}")
    print(f"Netteté min identification : {NETTETE_MIN_IDENTIF}")
