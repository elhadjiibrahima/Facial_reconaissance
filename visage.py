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

__version__ = "1.0.0"

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
SEUIL_RECONNAISSANCE = 0.363  # similarité cosinus recommandée pour SFace
SEUIL_PRECIS = 0.75           # score minimal des visages trouvés sur l'image agrandie
TAILLE_MIN_VISAGE = 60        # px : taille minimale pour enregistrer une personne
TAILLE_MIN_IDENTIF = 32       # px : en dessous, on ne tente pas d'identifier
NETTETE_MIN = 40.0            # variance du laplacien : en dessous, image floue


# =========================================================================== utilitaires
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def nettete(image) -> float:
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
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
               seuil: float = SEUIL_RECONNAISSANCE) -> tuple[str, float]:
    """Compare une empreinte normalisée aux personnes connues (similarité cosinus)."""
    meilleur, score = "Inconnu", -1.0
    for nom, emps in connus.items():
        s = float(np.max(emps @ empreinte))
        if s > score:
            meilleur, score = nom, s
    return (meilleur, score) if score >= seuil else ("Inconnu", score)


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
            if min(boite[2], boite[3]) < TAILLE_MIN_IDENTIF:
                nom, sim = "Inconnu", 0.0  # trop petit pour identifier de façon fiable
            else:
                nom, sim = identifier(emp, self.connus, self.seuil)
            resultats.append(Visage(
                boite, nom, sim,
                probas=self.probas_emotions(aligne) if self.emo is not None else None,
                vivant=self.probabilite_vivant(image, boite) if self.fas is not None else None))
        return resultats


# =========================================================================== suivi
@dataclass
class Piste:
    id: int
    boite: tuple
    probas: np.ndarray | None
    votes: Counter = field(default_factory=Counter)
    vue: float = 0.0


class Suivi:
    """Suit les visages d'une image à l'autre pour lisser l'émotion et le nom affichés."""

    def __init__(self, lissage: float = 0.6, iou_min: float = 0.3, oubli: float = 1.5):
        self.lissage, self.iou_min, self.oubli = lissage, iou_min, oubli
        self.pistes: dict[int, Piste] = {}
        self._prochain = 1

    def mettre_a_jour(self, visages: list[Visage], maintenant: float | None = None) -> list[Visage]:
        maintenant = time.time() if maintenant is None else maintenant
        libres = set(self.pistes)
        for v in sorted(visages, key=lambda v: -v.boite[2] * v.boite[3]):
            meilleure, score = None, self.iou_min
            for pid in libres:
                s = iou(v.boite, self.pistes[pid].boite)
                if s >= score:
                    meilleure, score = pid, s
            if meilleure is None:
                p = Piste(self._prochain, v.boite, None if v.probas is None else v.probas.copy())
                self.pistes[p.id] = p
                self._prochain += 1
            else:
                p = self.pistes[meilleure]
                libres.discard(meilleure)
                p.boite = v.boite
                if v.probas is not None and p.probas is not None:
                    p.probas = self.lissage * p.probas + (1 - self.lissage) * v.probas
            p.vue = maintenant
            p.votes[v.nom] += 1
            # le nom affiché = le plus souvent reconnu, en préférant un nom connu
            connus = [(n, c) for n, c in p.votes.most_common() if n != "Inconnu"]
            if connus and connus[0][1] * 2 >= p.votes["Inconnu"]:
                v.nom = connus[0][0]
            v.probas, v.piste = p.probas, p.id
        for pid in [pid for pid, p in self.pistes.items() if maintenant - p.vue > self.oubli]:
            del self.pistes[pid]
        return visages


# =========================================================================== stockage
def charger_visages() -> dict[str, np.ndarray]:
    connus = {}
    if VISAGES.exists():
        for f in sorted(VISAGES.glob("*.npy")):
            connus[f.stem] = np.load(f)
    return connus


def sauver_empreintes(nom: str, empreintes) -> int:
    VISAGES.mkdir(parents=True, exist_ok=True)
    fichier = VISAGES / f"{nom}.npy"
    nouv = np.stack(empreintes).astype(np.float32)
    if fichier.exists():
        nouv = np.vstack([np.load(fichier), nouv])
    np.save(fichier, nouv)
    return len(nouv)


# =========================================================================== dessin
def dessiner(image, visages: list[Visage], fps: float | None = None):
    for v in visages:
        x, y, w, h = v.boite
        couleur = COULEURS.get(v.emotion, (255, 255, 255))
        cv2.rectangle(image, (x, y), (x + w, y + h), couleur, 2)
        texte = v.nom if v.probas is None else f"{v.nom} - {v.emotion} {int(v.confiance * 100)}%"
        (tw, th), _ = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_txt = max(y - 8, th + 8)
        cv2.rectangle(image, (x, y_txt - th - 6), (x + tw + 6, y_txt + 4), couleur, -1)
        cv2.putText(image, texte, (x + 3, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    if fps is not None:
        cv2.putText(image, f"{fps:.0f} img/s  |  {len(visages)} visage(s)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return image


def ouvrir_source(camera: int, video: str | None):
    source = video if video else camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir {'la vidéo ' + video if video else f'la webcam n°{camera} (essayez --camera 1)'}")
    return cap


# =========================================================================== commandes
def cmd_enregistrer(args):
    a = Analyseur(emotions=False)
    empreintes: list[np.ndarray] = []

    def accepter(image, visage, origine="") -> str | None:
        """Renvoie la raison du refus, ou None si l'empreinte est gardée."""
        x, y, w, h = (int(t) for t in visage[:4])
        if min(w, h) < TAILLE_MIN_VISAGE:
            return "visage trop petit (rapprochez-vous)"
        emp, aligne = a.empreinte(image, visage)
        if nettete(aligne) < NETTETE_MIN:
            return "image floue"
        if empreintes and float(np.max(np.stack(empreintes) @ emp)) > 0.97:
            return "trop semblable à une capture précédente (bougez un peu)"
        # avertit si ce visage ressemble déjà fortement à une autre personne enregistrée
        autres = {n: e for n, e in a.connus.items() if n != args.nom}
        nom, sim = identifier(emp, autres, 0.5)
        if nom != "Inconnu" and origine:
            print(f"  ⚠ {origine} ressemble beaucoup à « {nom} » ({sim:.2f})")
        empreintes.append(emp)
        return None

    if args.images:
        for chemin in args.images:
            img = cv2.imread(chemin)
            if img is None:
                print(f"  ✘ {chemin} : image illisible"); continue
            visages = a.detecter(img)
            if len(visages) != 1:
                print(f"  ✘ {chemin} : {len(visages)} visage(s), il en faut exactement 1"); continue
            refus = accepter(img, visages[0], chemin)
            print(f"  ✘ {chemin} : {refus}" if refus else f"  ✔ {chemin}")
    else:
        cam = ouvrir_source(args.camera, None)
        print(f"Regardez la caméra et tournez doucement la tête. {args.nombre} captures… (q = annuler)")
        derniere, conseil = 0.0, ""
        while len(empreintes) < args.nombre:
            ok, frame = cam.read()
            if not ok:
                break
            visages = a.detecter(frame)
            affichage = frame.copy()
            for v in visages:
                x, y, w, h = (int(t) for t in v[:4])
                cv2.rectangle(affichage, (x, y), (x + w, y + h), (0, 200, 0), 2)
            if len(visages) > 1:
                conseil = "Une seule personne devant la camera !"
            elif len(visages) == 1 and time.time() - derniere > 0.4:
                refus = accepter(frame, visages[0])
                conseil = refus or ""
                if not refus:
                    derniere = time.time()
            texte = f"{len(empreintes)}/{args.nombre}  {conseil}"
            cv2.putText(affichage, texte, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("Enregistrement (q = annuler)", affichage)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cam.release()
        cv2.destroyAllWindows()
    if not empreintes:
        sys.exit("Aucune empreinte enregistrée.")
    total = sauver_empreintes(args.nom, empreintes)
    print(f"✔ {args.nom} : {len(empreintes)} empreinte(s) ajoutée(s), {total} au total.")


def cmd_lancer(args):
    a = Analyseur(seuil=args.seuil)
    if not a.connus:
        print("Personne n'est enregistré : tout le monde sera « Inconnu ».\n"
              "  Enregistrez-vous d'abord : python visage.py enregistrer VOTRE_NOM")
    cap = ouvrir_source(args.camera, args.video)
    suivi = Suivi()
    ecrivain_video = None
    journal = None
    if args.journal:
        nouveau = not Path(args.journal).exists() or Path(args.journal).stat().st_size == 0
        journal = open(args.journal, "a", newline="", encoding="utf-8")
        ecrivain = csv.writer(journal)
        if nouveau:
            ecrivain.writerow(["date", "nom", "emotion", "confiance"])
    dernier_log, t_prec, fps = 0.0, time.time(), 0.0
    resultats: list[Visage] = []
    n = 0
    print("Analyse lancée. q = quitter, s = capture d'écran.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.largeur and frame.shape[1] > args.largeur:
                r = args.largeur / frame.shape[1]
                frame = cv2.resize(frame, None, fx=r, fy=r)
            if n % (args.sauter + 1) == 0:
                resultats = suivi.mettre_a_jour(a.analyser(frame, args.precis))
            n += 1
            maintenant = time.time()
            fps = 0.9 * fps + 0.1 * (1 / max(maintenant - t_prec, 1e-6))
            t_prec = maintenant
            dessiner(frame, resultats, fps)

            if journal and resultats and maintenant - dernier_log >= args.intervalle:
                horodatage = datetime.now().isoformat(timespec="seconds")
                for v in resultats:
                    ecrivain.writerow([horodatage, v.nom, v.emotion, f"{v.confiance:.2f}"])
                journal.flush()
                dernier_log = maintenant

            if args.sortie:
                if ecrivain_video is None:
                    h, w = frame.shape[:2]
                    ips = cap.get(cv2.CAP_PROP_FPS) or 20
                    ecrivain_video = cv2.VideoWriter(
                        args.sortie, cv2.VideoWriter_fourcc(*"mp4v"), ips, (w, h))
                ecrivain_video.write(frame)

            if not args.sans_fenetre:
                cv2.imshow("Reconnaissance faciale et emotions (q = quitter)", frame)
                touche = cv2.waitKey(1) & 0xFF
                if touche == ord("q"):
                    break
                if touche == ord("s"):
                    nom = f"capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(nom, frame)
                    print(f"Capture enregistrée : {nom}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if ecrivain_video is not None:
            ecrivain_video.release()
            print(f"Vidéo annotée : {args.sortie}")
        if journal:
            journal.close()
            print(f"Journal des humeurs : {args.journal}  (résumé : python visage.py rapport {args.journal})")
        cv2.destroyAllWindows()


def cmd_photo(args):
    a = Analyseur(seuil=args.seuil)
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Image illisible : {args.image}")
    resultats = a.analyser(img, precis=not args.rapide)
    if not resultats:
        print("Aucun visage détecté.")
    for i, v in enumerate(resultats, 1):
        details = ", ".join(f"{e} {p * 100:.0f}%" for e, p in
                            sorted(zip(EMOTIONS, v.probas), key=lambda t: -t[1])[:3])
        qui = v.nom if v.nom == "Inconnu" else f"{v.nom} (ressemblance {v.similarite:.2f})"
        print(f"Visage {i} : {qui} — {details}")
    sortie = args.sortie or str(Path(args.image).with_name(Path(args.image).stem + "_analyse.jpg"))
    cv2.imwrite(sortie, dessiner(img, resultats))
    print(f"Image annotée : {sortie}")


def resumer_journal(chemin: str) -> dict[str, Counter]:
    par_personne: dict[str, Counter] = defaultdict(Counter)
    with open(chemin, encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            par_personne[ligne["nom"]][ligne["emotion"]] += 1
    return dict(par_personne)


def cmd_rapport(args):
    if not Path(args.journal).exists():
        sys.exit(f"Fichier introuvable : {args.journal}")
    resume = resumer_journal(args.journal)
    if not resume:
        print("Le journal est vide."); return
    for nom, compte in sorted(resume.items()):
        total = sum(compte.values())
        print(f"\n{nom} — {total} observation(s), humeur dominante : {compte.most_common(1)[0][0]}")
        for emo, c in compte.most_common():
            barre = "█" * round(c * 30 / total)
            print(f"  {emo:<10} {barre} {c * 100 / total:.0f}%")


def cmd_liste(_args):
    connus = charger_visages()
    if not connus:
        print("Aucune personne enregistrée.")
    for nom, emps in connus.items():
        print(f"- {nom} ({len(emps)} empreintes)")


def cmd_supprimer(args):
    f = VISAGES / f"{args.nom}.npy"
    if f.exists():
        f.unlink(); print(f"{args.nom} supprimé.")
    else:
        print(f"{args.nom} n'est pas enregistré.")


def construire_parseur() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reconnaissance faciale et détection des émotions")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sp = p.add_subparsers(dest="commande", required=True)

    e = sp.add_parser("enregistrer", help="enregistrer le visage d'une personne")
    e.add_argument("nom", type=nom_valide)
    e.add_argument("--images", nargs="+", help="photos de la personne (sinon : webcam)")
    e.add_argument("--nombre", type=int, default=15, help="captures webcam (15)")
    e.add_argument("--camera", type=int, default=0)
    e.set_defaults(func=cmd_enregistrer)

    l = sp.add_parser("lancer", help="analyse en direct (webcam ou vidéo)")
    l.add_argument("--camera", type=int, default=0, help="numéro de la webcam (0)")
    l.add_argument("--video", help="analyser un fichier vidéo au lieu de la webcam")
    l.add_argument("--sortie", help="enregistrer la vidéo annotée (ex : resultat.mp4)")
    l.add_argument("--journal", help="fichier CSV où noter les humeurs (ex : humeurs.csv)")
    l.add_argument("--intervalle", type=float, default=5, help="secondes entre deux notes du journal (5)")
    l.add_argument("--largeur", type=int, default=960, help="réduit l'image à cette largeur pour aller plus vite (960, 0 = non)")
    l.add_argument("--sauter", type=int, default=0, help="n'analyser qu'une image sur N+1 (PC lent : 1 ou 2)")
    l.add_argument("--seuil", type=float, default=SEUIL_RECONNAISSANCE, help="seuil de reconnaissance (0.363)")
    l.add_argument("--precis", action="store_true", help="trouver aussi les petits visages lointains (plus lent)")
    l.add_argument("--sans-fenetre", action="store_true", help="ne pas afficher de fenêtre (avec --sortie ou --journal)")
    l.set_defaults(func=cmd_lancer)

    ph = sp.add_parser("photo", help="analyser une photo")
    ph.add_argument("image")
    ph.add_argument("--sortie")
    ph.add_argument("--rapide", action="store_true", help="ne pas chercher les petits visages au fond")
    ph.add_argument("--seuil", type=float, default=SEUIL_RECONNAISSANCE)
    ph.set_defaults(func=cmd_photo)

    r = sp.add_parser("rapport", help="résumé des humeurs d'un journal CSV")
    r.add_argument("journal")
    r.set_defaults(func=cmd_rapport)

    sp.add_parser("liste", help="personnes enregistrées").set_defaults(func=cmd_liste)
    s = sp.add_parser("supprimer", help="supprimer une personne")
    s.add_argument("nom", type=nom_valide)
    s.set_defaults(func=cmd_supprimer)
    return p


def main(argv=None):
    args = construire_parseur().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
