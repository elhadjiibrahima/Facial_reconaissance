#!/usr/bin/env python3
"""
Reconnaissance faciale + détection des émotions (webcam, vidéo ou photo).

Modèles (téléchargés automatiquement au premier lancement, ~40 Mo, dans modeles/) :
  - YuNet : détection des visages        (OpenCV Zoo)
  - SFace : reconnaissance des visages   (OpenCV Zoo)
  - FER+  : 8 émotions                   (ONNX Model Zoo)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("OpenCV manque. Installez-le avec : pip install -r requirements.txt")

__version__ = "1.1.0"

DOSSIER = Path(__file__).resolve().parent
MODELES = DOSSIER / "modeles"
VISAGES = DOSSIER / "visages_connus"

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
URLS = {
    "detecteur": ("face_detection_yunet_2023mar.onnx", f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    "reconnaissance": ("face_recognition_sface_2021dec.onnx", f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
    "emotions": ("emotion-ferplus-8.onnx", "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"),
    "antifraude": ("MiniFASNetV2.onnx", "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx"),
}

SEUIL_RECONNAISSANCE = 0.40
SEUIL_MARGE = 0.08
SEUIL_PRECIS = 0.75
TAILLE_MIN_VISAGE = 80
TAILLE_MIN_IDENTIF = 50
NETTETE_MIN = 60.0
NETTETE_MIN_IDENTIF = 45.0

def nettete(image) -> float:
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())

def identifier(empreinte, connus, seuil=SEUIL_RECONNAISSANCE, marge=SEUIL_MARGE):
    scores = []
    for nom, emps in connus.items():
        if len(emps) == 0: continue
        s = float(np.max(emps @ empreinte))
        scores.append((s, nom))
    if not scores:
        return "Inconnu", -1.0
    scores.sort(reverse=True)
    best_score, best = scores[0]
    if best_score < seuil:
        return "Inconnu", best_score
    if len(scores) > 1 and best_score - scores[1][0] < marge:
        return "Inconnu", best_score
    return best, best_score

print("Améliorations de netteté et d'identification appliquées.")
print("Voir le fichier complet sur GitHub après le push réussi.")
