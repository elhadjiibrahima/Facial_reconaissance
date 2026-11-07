#!/usr/bin/env python3
"""
Reconnaissance faciale + détection des émotions (webcam, vidéo ou photo).

Modèles (téléchargés automatiquement au premier lancement, ~40 Mo, dans modeles/) :
  - YuNet : détection des visages        (OpenCV Zoo)
  - SFace : reconnaissance des visages   (OpenCV Zoo)
  - FER+  : 8 émotions                   (ONNX Model Zoo)

Exemples :
  python visage.py enregistrer Ibrahima                  # captures via la webcam
  python visage.py enregistrer Awa --images a.jpg b.jpg  # à partir de photos
  python visage.py lancer                                # webcam en direct
  python visage.py lancer --video reunion.mp4 --sortie reunion_analyse.mp4
  python visage.py lancer --journal humeurs.csv          # note les humeurs dans un CSV
  python visage.py photo groupe.jpg                      # -> groupe_analyse.jpg
  python visage.py rapport humeurs.csv                   # résumé des humeurs par personne
  python visage.py liste | supprimer Awa
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
except ImportError:  # pragma: no cover
    sys.exit("OpenCV manque. Installez-le avec : pip install -r requirements.txt")

__version__ = "1.1.0"

DOSSIER = Path(__file__).resolve().parent
MODELES = DOSSIER / "modeles"
VISAGES = DOSSIER / "visages_connus"

# Les modèles sont stockés avec Git LFS : on passe directement par media.githubusercontent.com
ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
URLS = {
    "detecteur": ("face_detection_yunet_2023mar.onnx",
                  f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    "reconnaissance": ("face_recognition_sface_2021dec.onnx",
                       f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
    "emotions": ("emotion-ferplus-8.onnx",
                 "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/"
                 "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"),
    # anti-fraude : distingue un vrai visage d'une photo ou d'un écran (Silent-Face / MiniVision)
    "antifraude": ("MiniFASNetV2.onnx",
                   "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx"),
}
EMPREINTES_SHA256 = {
    "MiniFASNetV2.onnx": "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907",
}

# Ordre des sorties du modèle FER+
EMOTIONS = ["Neutre", "Joie", "Surprise", "Tristesse", "Colere", "Degout", "Peur", "Mepris"]
COULEURS = {  # BGR
    "Neutre": (200, 200, 200), "Joie": (0, 200, 0), "Surprise": (0, 200, 255),
    "Tristesse": (255, 120, 0), "Colere": (0, 0, 255), "Degout": (0, 140, 140),
    "Peur": (180, 0, 180), "Mepris": (120, 60, 160),
}
SEUIL_RECONNAISSANCE = 0.40   # un peu plus strict pour réduire les faux positifs
SEUIL_MARGE = 0.08            # différence minimale entre le 1er et le 2e meilleur match
SEUIL_PRECIS = 0.75           # score minimal des visages trouvés sur l'image agrandie
TAILLE_MIN_VISAGE = 80        # px : taille minimale pour enregistrer une personne
TAILLE_MIN_IDENTIF = 50       # px : en dessous, identification peu fiable
NETTETE_MIN = 60.0            # variance du laplacien (plus strict)
NETTETE_MIN_IDENTIF = 45.0    # netteté minimale pour tenter une identification


# =========================================================================== utilitaires
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def nettete(image) -> float:
    """Mesure la netteté via la variance du Laplacien (plus la valeur est haute, plus c'est net)."""
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Léger flou gaussien pour réduire le bruit avant le Laplacien
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nom_valide(nom: str) -> str:
    nom = nom.strip()
    if not nom or any(c in nom for c in '/\\:*?"<>|') or nom.startswith("."):
        raise argparse.ArgumentTypeError(f"nom invalide : {nom!r}")
    return nom


def identifier(empreinte: np.ndarray, connus: dict[str, np.ndarray],
               seuil: float = SEUIL_RECONNAISSANCE,
               marge: float = SEUIL_MARGE) -> tuple[str, float]:
    """Compare une empreinte normalisée aux personnes connues (similarité cosinus).

    Exige une marge suffisante avec le deuxième meilleur match pour éviter les confusions.
    """
    scores = []
    for nom, emps in connus.items():
        if len(emps) == 0:
            continue
        s = float(np.max(emps @ empreinte))
        scores.append((s, nom))
    if not scores:
        return "Inconnu", -1.0
    scores.sort(reverse=True)
    meilleur_score, meilleur = scores[0]
    if meilleur_score < seuil:
        return "Inconnu", meilleur_score
    # Vérifier la marge avec le 2e
    if len(scores) > 1:
        second = scores[1][0]
        if meilleur_score - second < marge:
            return "Inconnu", meilleur_score  # trop ambigu
    return meilleur, meilleur_score


# =========================================================================== modèles
def chemin_modele(cle: str) -> Path:
    nom, url = URLS[cle]
    chemin = MODELES / nom
    if chemin.exists() and chemin.stat().st_size > 100_000:
        return chemin
    MODELES.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_suffix(".part")

    dernier_pct = [-1]

    def progression(blocs, taille_bloc, total):
        if total > 0:
            pct = min(100, blocs * taille_bloc * 100 // total) // 10 * 10
            if pct != dernier_pct[0]:
                dernier_pct[0] = pct
                print(f"\r  {nom} : {pct}%", end="", flush=True)

    print(f"Téléchargement du modèle {nom}…")
    try:
        requete = urllib.request.Request(url, headers={"User-Agent": f"Facial_reconaissance/{__version__}"})
        with urllib.request.urlopen(requete, timeout=60) as rep, open(tmp, "wb") as f:
            total = int(rep.headers.get("Content-Length") or 0)
            blocs = 0
            while morceau := rep.read(64 * 1024):
                f.write(morceau)
                blocs += 1
                progression(blocs, 64 * 1024, total)
        print()
    except Exception as e:
        tmp.unlink(missing_ok=True)
        sys.exit(f"\nImpossible de télécharger {url}\n  ({e})\n"
                 f"Téléchargez-le à la main et placez-le dans {MODELES}/")
    attendu = EMPREINTES_SHA256.get(nom)
    if tmp.stat().st_size < 100_000 or (attendu and hashlib.sha256(tmp.read_bytes()).hexdigest() != attendu):
        tmp.unlink()  # pointeur Git LFS, page d'erreur ou fichier modifié
        sys.exit(f"Fichier reçu invalide pour {nom}. Téléchargez-le à la main :\n  {url}")
    tmp.replace(chemin)
    return chemin


@dataclass
class Visage:
    boite: tuple[int, int, int, int]
    nom: str
    similarite: float
    probas: np.ndarray | None = None   # émotions (None si désactivées)
    vivant: float | None = None        # probabilité d'un vrai visage (None si anti-fraude désactivée)
    piste: int = -1

    @property
    def emotion(self) -> str:
        return "" if self.probas is None else EMOTIONS[int(np.argmax(self.probas))]

    @property
    def confiance(self) -> float:
        return 0.0 if self.probas is None else float(np.max(self.probas))


class Analyseur:
    def __init__(self, seuil: float = SEUIL_RECONNAISSANCE, seuil_detection: float = 0.8,
                 emotions: bool = True, antifraude: bool = False):
        self.seuil = seuil
        self.seuil_detection = seuil_detection
        self.detecteur = cv2.FaceDetectorYN.create(
            str(chemin_modele("detecteur")), "", (320, 320),
            min(seuil_detection, SEUIL_PRECIS), 0.3, 5000)
        self.reco = cv2.FaceRecognizerSF.create(str(chemin_modele("reconnaissance")), "")
        self.emo = cv2.dnn.readNetFromONNX(str(chemin_modele("emotions"))) if emotions else None
        self.fas = cv2.dnn.readNetFromONNX(str(chemin_modele("antifraude"))) if antifraude else None
        self.connus = charger_visages()

    def recharger_visages(self):
        self.connus = charger_visages()

    def _detecter_echelle(self, image, echelle: float):
        img = image if echelle == 1 else cv2.resize(image, None, fx=echelle, fy=echelle)
        h, w = img.shape[:2]
        self.detecteur.setInputSize((w, h))
        _, visages = self.detecteur.detect(img)
        if visages is None:
            return []
        visages = visages.copy()
        visages[:, :14] /= echelle  # boîte + 5 points du visage, ramenés à l'image d'origine
        return list(visages)

    def detecter(self, image, precis: bool = False):
        """precis=True : cherche aussi sur l'image agrandie x2 pour trouver les petits visages
        (personnes au fond d'une photo). Plus lent : à utiliser surtout sur les photos."""
        visages = [v for v in self._detecter_echelle(image, 1.0) if v[14] >= self.seuil_detection]
        if precis and max(image.shape[:2]) * 2 <= 4000:
            visages += [v for v in self._detecter_echelle(image, 2.0) if v[14] >= SEUIL_PRECIS]
            if len(visages) > 1:
                boites = [[float(x) for x in v[:4]] for v in visages]
                scores = [float(v[14]) for v in visages]
                garder = cv2.dnn.NMSBoxes(boites, scores, 0.0, 0.3)
                visages = [visages[i] for i in np.array(garder).flatten()]
        return visages

    def empreinte(self, image, visage):
        aligne = self.reco.alignCrop(image, visage)  # visage recadré et redressé, 112x112
        f = self.reco.feature(aligne).flatten().astype(np.float32)
        return f / (np.linalg.norm(f) + 1e-9), aligne

    def probas_emotions(self, visage_aligne) -> np.ndarray:
        gris = cv2.cvtColor(visage_aligne, cv2.COLOR_BGR2GRAY)
        gris = cv2.resize(gris, (64, 64)).astype(np.float32)
        self.emo.setInput(gris.reshape(1, 1, 64, 64))
        return softmax(self.emo.forward().flatten())

    def probabilite_vivant(self, image, boite, echelle: float = 2.7) -> float:
        """Probabilité que le visage soit réel (et non une photo, un écran ou une vidéo)."""
        hi, wi = image.shape[:2]
        x, y, w, h = boite
        if w <= 0 or h <= 0:
            return 0.0
        s = min((hi - 1) / h, (wi - 1) / w, echelle)
        cx, cy, nw, nh = x + w / 2, y + h / 2, w * s, h * s
        x1, y1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
        x2, y2 = min(wi - 1, int(cx + nw / 2)), min(hi - 1, int(cy + nh / 2))
        zone = cv2.resize(image[y1:y2 + 1, x1:x2 + 1], (80, 80)).astype(np.float32)
        self.fas.setInput(zone.transpose(2, 0, 1)[None])
        return float(softmax(self.fas.forward().flatten())[1])  # classe 1 = vrai visage

    def analyser(self, image, precis: bool = False) -> list[Visage]:
        resultats = []
        for v in self.detecter(image, precis):
            boite = tuple(int(x) for x in v[:4])
            emp, aligne = self.empreinte(image, v)
            taille = min(boite[2], boite[3])
            score_nettete = nettete(aligne)

            # Trop petit ou trop flou → pas d'identification fiable
            if taille < TAILLE_MIN_IDENTIF or score_nettete < NETTETE_MIN_IDENTIF:
                nom, sim = "Inconnu", 0.0
            else:
                nom, sim = identifier(emp, self.connus, self.seuil)
            resultats.append(Visage(
                boite, nom, sim,
                probas=self.probas_emotions(aligne) if self.emo is not None else None,
                vivant=self.probabilite_vivant(image, boite) if self.fas is not None else None))
        return resultats
