#!/usr/bin/env python3
"""
Pointage des employés par reconnaissance faciale.

Une « borne » (webcam, caméra IP ou caméra de surveillance en RTSP) reconnaît les employés
et enregistre leurs passages : le premier de la journée = arrivée, le dernier = départ.
Une photo du visage est gardée comme preuve, et un modèle anti-fraude refuse les photos
et les écrans présentés devant la caméra.

Exemples :
  python pointage.py employe ajouter "Awa Diallo" --matricule E001 --service Compta --debut 08:00 --fin 17:00
  python pointage.py employe ajouter "Moussa Bah" --images moussa1.jpg moussa2.jpg
  python pointage.py borne                                   # webcam
  python pointage.py borne --source "rtsp://admin:mdp@192.168.1.64:554/Streaming/Channels/101" --nom-borne Entree
  python pointage.py aujourdhui                              # qui est là, qui est en retard, qui manque
  python pointage.py rapport --mois 2026-11 --csv novembre.csv
  python pointage.py historique "Awa Diallo"
  python pointage.py corriger "Awa Diallo" --le "2026-11-07 08:05" --motif "panne de la caméra"
  python pointage.py annuler 42 --motif "doublon"
  python pointage.py alertes                                 # tentatives de fraude
"""
from __future__ import annotations

import argparse
import csv
import queue
import re
import sqlite3
import sys
import threading
import time
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

import visage as vz

cv2 = vz.cv2

BASE = vz.DOSSIER / "pointage.db"
PHOTOS = vz.DOSSIER / "photos_pointage"

SEUIL_POINTAGE = 0.40      # plus strict que la reconnaissance simple : on enregistre de vraies présences
SEUIL_VIVANT = 0.80        # probabilité moyenne minimale d'un vrai visage
IMAGES_CONFIRMATION = 5    # nombre d'images où la même personne doit être reconnue
DELAI_DOUBLON = 120        # secondes : un 2e passage plus tôt est ignoré
TOLERANCE_RETARD = 5       # minutes

SCHEMA = """
CREATE TABLE IF NOT EXISTS employes(
    id INTEGER PRIMARY KEY,
    nom TEXT NOT NULL UNIQUE,
    matricule TEXT UNIQUE,
    service TEXT NOT NULL DEFAULT '',
    heure_debut TEXT NOT NULL DEFAULT '08:00',
    heure_fin TEXT NOT NULL DEFAULT '17:00',
    actif INTEGER NOT NULL DEFAULT 1,
    cree_le TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pointages(
    id INTEGER PRIMARY KEY,
    employe_id INTEGER NOT NULL REFERENCES employes(id),
    horodatage TEXT NOT NULL,
    borne TEXT NOT NULL DEFAULT '',
    ressemblance REAL,
    vivant REAL,
    photo TEXT,
    manuel INTEGER NOT NULL DEFAULT 0,
    motif TEXT,
    annule INTEGER NOT NULL DEFAULT 0,
    motif_annulation TEXT
);
CREATE INDEX IF NOT EXISTS idx_pointages ON pointages(employe_id, horodatage);
CREATE TABLE IF NOT EXISTS alertes(
    id INTEGER PRIMARY KEY,
    horodatage TEXT NOT NULL,
    borne TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL,
    nom_suppose TEXT,
    score REAL,
    photo TEXT
);
"""


# =========================================================================== utilitaires
def heure_valide(texte: str) -> str:
    m = re.fullmatch(r"([01]?\d|2[0-3])[:hH]([0-5]\d)", texte.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"heure invalide : {texte!r} (format 08:00)")
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def jours_ouvres(texte: str) -> set[int]:
    """'1-5' -> lundi à vendredi ; '1-6' -> lundi à samedi ; '1,2,3,4,5'. (1 = lundi)"""
    jours: set[int] = set()
    for morceau in texte.split(","):
        if "-" in morceau:
            a, b = (int(x) for x in morceau.split("-"))
            jours.update(range(a, b + 1))
        elif morceau.strip():
            jours.add(int(morceau))
    if not jours or not jours <= set(range(1, 8)):
        raise argparse.ArgumentTypeError("jours ouvrés invalides (ex : 1-5 pour lundi à vendredi)")
    return jours


def a_la_minute(d: datetime) -> str:
    return d.strftime("%H:%M")


def duree_texte(minutes: float | None) -> str:
    if minutes is None:
        return ""
    minutes = int(round(minutes))
    return f"{minutes // 60}h{minutes % 60:02d}"


def sans_accents(texte: str) -> str:
    return unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode()


# =========================================================================== registre
@dataclass
class Employe:
    id: int
    nom: str
    matricule: str | None
    service: str
    heure_debut: str
    heure_fin: str
    actif: bool
    cree_le: str


@dataclass
class Journee:
    jour: date
    passages: list[datetime]
    arrivee: datetime | None = None
    depart: datetime | None = None
    retard: int = 0              # minutes après l'heure de début (au-delà de la tolérance)
    depart_anticipe: int = 0     # minutes avant l'heure de fin
    duree: float | None = None   # minutes de présence

    @property
    def statut(self) -> str:
        if not self.passages:
            return "Absent"
        if self.depart is None:
            return "Présent (pas de départ)"
        return "Présent"


class Registre:
    def __init__(self, chemin: Path | str = BASE):
        self.db = sqlite3.connect(str(chemin), timeout=15, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")   # plusieurs bornes peuvent écrire en même temps
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    # ---------------------------------------------------------------- employés
    @staticmethod
    def _employe(ligne) -> Employe:
        return Employe(ligne["id"], ligne["nom"], ligne["matricule"], ligne["service"],
                       ligne["heure_debut"], ligne["heure_fin"], bool(ligne["actif"]), ligne["cree_le"])

    def ajouter_employe(self, nom: str, matricule: str | None = None, service: str = "",
                        debut: str = "08:00", fin: str = "17:00", quand: datetime | None = None) -> Employe:
        quand = quand or datetime.now()
        try:
            with self.db:
                self.db.execute(
                    "INSERT INTO employes(nom, matricule, service, heure_debut, heure_fin, cree_le) "
                    "VALUES (?,?,?,?,?,?)",
                    (nom, matricule or None, service, debut, fin, quand.isoformat(timespec="seconds")))
        except sqlite3.IntegrityError:
            raise ValueError(f"« {nom} » ou le matricule {matricule} existe déjà") from None
        return self.employe(nom)

    def employe(self, nom_ou_matricule: str) -> Employe | None:
        ligne = self.db.execute("SELECT * FROM employes WHERE nom = ? OR matricule = ?",
                                (nom_ou_matricule, nom_ou_matricule)).fetchone()
        return self._employe(ligne) if ligne else None

    def employes(self, actifs: bool = False) -> list[Employe]:
        req = "SELECT * FROM employes" + (" WHERE actif = 1" if actifs else "") + " ORDER BY nom"
        return [self._employe(l) for l in self.db.execute(req)]

    def modifier_employe(self, nom: str, **champs) -> Employe:
        autorises = {"matricule", "service", "heure_debut", "heure_fin", "actif"}
        champs = {k: v for k, v in champs.items() if k in autorises and v is not None}
        e = self.employe(nom)
        if e is None:
            raise ValueError(f"Employé inconnu : {nom}")
        if champs:
            with self.db:
                self.db.execute(
                    f"UPDATE employes SET {', '.join(f'{k} = ?' for k in champs)} WHERE id = ?",
                    (*champs.values(), e.id))
        return self.employe(e.nom)

    # ---------------------------------------------------------------- pointages
    def pointer(self, employe: Employe, quand: datetime, borne: str = "", ressemblance: float | None = None,
                vivant: float | None = None, photo: str | None = None, motif: str | None = None) -> int:
        with self.db:
            cur = self.db.execute(
                "INSERT INTO pointages(employe_id, horodatage, borne, ressemblance, vivant, photo, manuel, motif) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (employe.id, quand.isoformat(timespec="seconds"), borne, ressemblance, vivant, photo,
                 1 if motif else 0, motif))
        return cur.lastrowid

    def annuler(self, pointage_id: int, motif: str) -> bool:
        with self.db:
            cur = self.db.execute("UPDATE pointages SET annule = 1, motif_annulation = ? "
                                  "WHERE id = ? AND annule = 0", (motif, pointage_id))
        return cur.rowcount == 1

    def passages(self, employe: Employe, du: date, au: date) -> list[datetime]:
        lignes = self.db.execute(
            "SELECT horodatage FROM pointages WHERE employe_id = ? AND annule = 0 "
            "AND horodatage >= ? AND horodatage < ? ORDER BY horodatage",
            (employe.id, du.isoformat(), (au + timedelta(days=1)).isoformat()))
        return [datetime.fromisoformat(l["horodatage"]) for l in lignes]

    def historique(self, employe: Employe, du: date, au: date):
        return self.db.execute(
            "SELECT * FROM pointages WHERE employe_id = ? AND horodatage >= ? AND horodatage < ? "
            "ORDER BY horodatage", (employe.id, du.isoformat(), (au + timedelta(days=1)).isoformat())
        ).fetchall()

    def alerte(self, type_: str, quand: datetime, borne: str = "", nom_suppose: str | None = None,
               score: float | None = None, photo: str | None = None):
        with self.db:
            self.db.execute("INSERT INTO alertes(horodatage, borne, type, nom_suppose, score, photo) "
                            "VALUES (?,?,?,?,?,?)",
                            (quand.isoformat(timespec="seconds"), borne, type_, nom_suppose, score, photo))

    def alertes(self, depuis: date):
        return self.db.execute("SELECT * FROM alertes WHERE horodatage >= ? ORDER BY horodatage DESC",
                               (depuis.isoformat(),)).fetchall()

    # ---------------------------------------------------------------- calculs
    def journee(self, employe: Employe, jour: date, tolerance: int = TOLERANCE_RETARD) -> Journee:
        p = self.passages(employe, jour, jour)
        j = Journee(jour, p)
        if not p:
            return j
        j.arrivee = p[0]
        prevu_debut = datetime.combine(jour, datetime.strptime(employe.heure_debut, "%H:%M").time())
        prevu_fin = datetime.combine(jour, datetime.strptime(employe.heure_fin, "%H:%M").time())
        retard = (j.arrivee - prevu_debut).total_seconds() / 60
        j.retard = int(retard) if retard > tolerance else 0
        if len(p) > 1:
            j.depart = p[-1]
            j.duree = (j.depart - j.arrivee).total_seconds() / 60
            avance = (prevu_fin - j.depart).total_seconds() / 60
            j.depart_anticipe = int(avance) if avance > tolerance else 0
        return j

    def type_prochain_passage(self, employe: Employe, quand: datetime) -> tuple[str, datetime | None]:
        """('arrivee', None), ('depart', None) ou ('doublon', heure du passage précédent)."""
        du_jour = self.passages(employe, quand.date(), quand.date())
        if du_jour and (quand - du_jour[-1]).total_seconds() < DELAI_DOUBLON:
            return "doublon", du_jour[-1]
        return ("depart" if du_jour else "arrivee"), None

    def rapport(self, du: date, au: date, ouvres: set[int], tolerance: int = TOLERANCE_RETARD,
                aujourd_hui: date | None = None):
        """Une ligne par employé et par jour ouvré (ou jour avec passage), plus un résumé par employé."""
        aujourd_hui = aujourd_hui or date.today()
        lignes, resume = [], []
        for e in self.employes():
            embauche = datetime.fromisoformat(e.cree_le).date()
            stats = Counter()
            minutes = 0.0
            jour = du
            while jour <= min(au, aujourd_hui):
                j = self.journee(e, jour, tolerance)
                ouvre = jour.isoweekday() in ouvres and jour >= embauche and (e.actif or j.passages)
                if j.passages or ouvre:
                    lignes.append((e, j))
                    if j.passages:
                        stats["presents"] += 1
                        stats["retards"] += j.retard > 0
                        stats["minutes_retard"] += j.retard
                        stats["sans_depart"] += j.depart is None
                        minutes += j.duree or 0
                    elif ouvre and jour < aujourd_hui:
                        stats["absences"] += 1
                    if ouvre:
                        stats["ouvres"] += 1
                jour += timedelta(days=1)
            resume.append((e, stats, minutes))
        return lignes, resume


# =========================================================================== décision à la borne
@dataclass
class Decision:
    type: str                 # 'reconnu', 'fraude', 'inconnu'
    nom: str = ""
    ressemblance: float = 0.0
    vivant: float | None = None


@dataclass
class EtatPiste:
    noms: deque = field(default_factory=lambda: deque(maxlen=IMAGES_CONFIRMATION + 3))
    sims: deque = field(default_factory=lambda: deque(maxlen=IMAGES_CONFIRMATION + 3))
    vivants: deque = field(default_factory=lambda: deque(maxlen=IMAGES_CONFIRMATION + 3))
    traite_a: float | None = None


class Validateur:
    """Ne valide une personne qu'après plusieurs images concordantes, et vérifie que c'est un vrai visage."""

    def __init__(self, images: int = IMAGES_CONFIRMATION, seuil_vivant: float | None = SEUIL_VIVANT,
                 reessai: float = 3.0):
        self.images, self.seuil_vivant, self.reessai = images, seuil_vivant, reessai
        self.pistes: dict[int, EtatPiste] = {}

    def observer(self, piste: int, nom: str, ressemblance: float, vivant: float | None,
                 maintenant: float) -> Decision | None:
        e = self.pistes.setdefault(piste, EtatPiste())
        if e.traite_a is not None:
            if e.traite_a == float("inf") or maintenant - e.traite_a < self.reessai:
                return None
            self.pistes[piste] = e = EtatPiste()  # nouvelle tentative (après refus)
        e.noms.append(nom)
        e.sims.append(ressemblance)
        if vivant is not None:
            e.vivants.append(vivant)
        if len(e.noms) < self.images:
            return None
        meilleur, nb = Counter(e.noms).most_common(1)[0]
        if nb < self.images:
            return None
        if meilleur == "Inconnu":
            e.traite_a = maintenant
            return Decision("inconnu")
        sims = [s for n, s in zip(e.noms, e.sims) if n == meilleur]
        v = float(np.mean(list(e.vivants)[-self.images:])) if e.vivants else None
        if self.seuil_vivant is not None and v is not None and v < self.seuil_vivant:
            e.traite_a = maintenant
            return Decision("fraude", meilleur, float(np.mean(sims)), v)
        e.traite_a = float("inf")  # pointé : on ne revalide plus cette piste
        return Decision("reconnu", meilleur, float(np.mean(sims)), v)

    def nettoyer(self, actives: set[int]):
        for pid in list(self.pistes):
            if pid not in actives:
                del self.pistes[pid]


# =========================================================================== affichage
class Ecrivain:
    """Écrit du texte avec accents (Pillow) ; sans Pillow, retombe sur OpenCV sans accents."""
    POLICES = ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
               "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"]

    def __init__(self):
        self.cache = {}
        try:
            from PIL import Image, ImageDraw, ImageFont
            self.Image, self.ImageDraw, self.ImageFont = Image, ImageDraw, ImageFont
            self.police = next((p for p in self.POLICES if Path(p).exists()), None)
        except ImportError:
            self.Image = None

    def _font(self, taille):
        if taille not in self.cache:
            if self.police:
                self.cache[taille] = self.ImageFont.truetype(self.police, taille)
            else:
                try:
                    self.cache[taille] = self.ImageFont.load_default(size=taille)
                except TypeError:
                    self.cache[taille] = self.ImageFont.load_default()
        return self.cache[taille]

    def ecrire(self, image, textes):
        """textes : liste de (texte, (x, y), taille, couleur BGR)."""
        if not textes:
            return image
        if self.Image is None:
            for t, (x, y), taille, c in textes:
                cv2.putText(image, sans_accents(t), (x, y + taille), cv2.FONT_HERSHEY_SIMPLEX,
                            taille / 32, c, 2)
            return image
        pil = self.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        d = self.ImageDraw.Draw(pil)
        for t, pos, taille, c in textes:
            d.text(pos, t, font=self._font(taille), fill=(c[2], c[1], c[0]))
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


VERT, ROUGE, BLANC, ORANGE, NOIR = (60, 180, 60), (40, 40, 220), (255, 255, 255), (0, 160, 255), (0, 0, 0)


# =========================================================================== lecture vidéo
class Source:
    """Webcam, fichier vidéo ou flux réseau (RTSP/HTTP). Pour un flux en direct, un fil séparé garde
    seulement la dernière image (pas de retard qui s'accumule) et se reconnecte en cas de coupure."""

    def __init__(self, source: str):
        self.source = int(source) if source.isdigit() else source
        self.direct = not (isinstance(self.source, str) and Path(self.source).exists())
        self.file: queue.Queue = queue.Queue(maxsize=1)
        self.actif = True
        self.cap = self._ouvrir()
        if self.direct:
            threading.Thread(target=self._boucle, daemon=True).start()

    def _ouvrir(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _boucle(self):
        attente = 1.0
        while self.actif:
            ok, image = self.cap.read() if self.cap.isOpened() else (False, None)
            if not ok:
                print(f"[{datetime.now():%H:%M:%S}] Caméra indisponible, nouvelle tentative dans {attente:.0f} s…")
                self.cap.release()
                time.sleep(attente)
                attente = min(attente * 2, 30)
                self.cap = self._ouvrir()
                continue
            attente = 1.0
            if self.file.full():
                try:
                    self.file.get_nowait()
                except queue.Empty:
                    pass
            self.file.put(image)

    def lire(self):
        if not self.direct:
            ok, image = self.cap.read()
            return image if ok else None
        try:
            return self.file.get(timeout=5)
        except queue.Empty:
            return np.zeros((480, 640, 3), np.uint8)  # écran noir pendant la reconnexion

    def fermer(self):
        self.actif = False
        self.cap.release()


def sauver_photo(image, boite, quand: datetime, etiquette: str) -> str:
    x, y, w, h = boite
    m = int(0.4 * max(w, h))
    hi, wi = image.shape[:2]
    zone = image[max(0, y - m):min(hi, y + h + m), max(0, x - m):min(wi, x + w + m)]
    dossier = PHOTOS / quand.strftime("%Y-%m-%d")
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"{quand:%H%M%S}_{sans_accents(etiquette).replace(' ', '_')}.jpg"
    cv2.imwrite(str(chemin), zone, [cv2.IMWRITE_JPEG_QUALITY, 85])
    try:
        return str(chemin.relative_to(vz.DOSSIER))
    except ValueError:
        return str(chemin)


def dessiner_borne(image, visages, couleurs: dict, messages: list, nom_borne: str, ecrivain: Ecrivain,
                   maintenant: datetime | None = None):
    maintenant = maintenant or datetime.now()
    textes = []
    for v in visages:
        x, y, w, h = v.boite
        c = couleurs.get(v.piste, BLANC)
        cv2.rectangle(image, (x, y), (x + w, y + h), c, 3)
        if v.nom != "Inconnu":
            y_nom = max(48, y - 34)
            cv2.rectangle(image, (x - 2, y_nom - 2), (x + 14 * len(v.nom) + 14, y_nom + 32), c, -1)
            textes.append((v.nom, (x + 6, y_nom + 2), 24, BLANC))
    hi, wi = image.shape[:2]
    cv2.rectangle(image, (0, 0), (wi, 44), NOIR, -1)
    textes.append((f"{nom_borne}   {maintenant:%d/%m/%Y  %H:%M:%S}", (12, 8), 24, BLANC))
    derniers = messages[-2:]
    for i, (texte, couleur, _) in enumerate(derniers):
        y0 = hi - 70 * (len(derniers) - i)
        cv2.rectangle(image, (0, y0), (wi, y0 + 64), couleur, -1)
        textes.append((texte, (16, y0 + 14), 32, BLANC))
    return ecrivain.ecrire(image, textes)


# =========================================================================== borne
def cmd_borne(args):
    reg = Registre(args.base)
    a = vz.Analyseur(seuil=args.seuil, emotions=False, antifraude=not args.sans_antifraude)
    actifs = {e.nom for e in reg.employes(actifs=True)}
    sans_visage = sorted(actifs - set(a.connus))
    if not actifs:
        print("Aucun employé enregistré. Commencez par : python pointage.py employe ajouter \"Prénom Nom\"")
    if sans_visage:
        print("Employés sans visage enregistré : " + ", ".join(sans_visage))
    if args.sans_antifraude:
        print("⚠ Anti-fraude désactivée : une photo de l'employé pourrait suffire à pointer.")

    source = Source(args.source)
    suivi = vz.Suivi(lissage=0.5)
    valid = Validateur(seuil_vivant=None if args.sans_antifraude else args.seuil_vivant)
    ecrivain = Ecrivain()
    messages: list[tuple[str, tuple, float]] = []  # (texte, couleur, expiration)
    derniere_maj = time.time()
    fenetre = f"Pointage - {args.nom_borne}"
    if not args.sans_fenetre:
        cv2.namedWindow(fenetre, cv2.WINDOW_NORMAL)
        if args.plein_ecran:
            cv2.setWindowProperty(fenetre, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def annoncer(texte, couleur):
        messages.append((texte, couleur, time.time() + 4))
        print(f"[{datetime.now():%H:%M:%S}] {texte}")

    print(f"Borne « {args.nom_borne} » prête. q = quitter.")
    try:
        while True:
            image = source.lire()
            if image is None:
                break
            if args.largeur and image.shape[1] > args.largeur:
                r = args.largeur / image.shape[1]
                image = cv2.resize(image, None, fx=r, fy=r)

            # nouveaux employés enregistrés pendant que la borne tourne
            if time.time() - derniere_maj > 30:
                a.recharger_visages()
                derniere_maj = time.time()

            visages = a.analyser(image, precis=args.precis)
            bruts = [(v.nom, v.similarite, v.vivant) for v in visages]
            suivi.mettre_a_jour(visages)
            maintenant = time.time()
            couleurs = {}
            for v, (nom, sim, vivant) in zip(visages, bruts):
                couleurs[v.piste] = VERT if v.nom != "Inconnu" else BLANC
                d = valid.observer(v.piste, nom, sim, vivant, maintenant)
                if d is None:
                    continue
                quand = datetime.now()
                if d.type == "inconnu":
                    annoncer("Visage non reconnu", ORANGE)
                    if args.photos_inconnus:
                        reg.alerte("inconnu", quand, args.nom_borne,
                                   photo=sauver_photo(image, v.boite, quand, "inconnu"))
                elif d.type == "fraude":
                    couleurs[v.piste] = ROUGE
                    annoncer("Pointage refusé : regardez la caméra (photo ou écran détecté ?)", ROUGE)
                    reg.alerte("fraude", quand, args.nom_borne, d.nom, d.vivant,
                               sauver_photo(image, v.boite, quand, f"fraude_{d.nom}"))
                else:
                    e = reg.employe(d.nom)
                    if e is None:
                        annoncer(f"{d.nom} n'est pas inscrit comme employé", ORANGE)
                    elif not e.actif:
                        annoncer(f"{e.nom} : compte désactivé", ORANGE)
                    else:
                        type_, avant = reg.type_prochain_passage(e, quand)
                        if type_ == "doublon":
                            annoncer(f"{e.nom} : déjà pointé à {a_la_minute(avant)}", VERT)
                        else:
                            photo = sauver_photo(image, v.boite, quand, e.nom)
                            reg.pointer(e, quand, args.nom_borne, d.ressemblance, d.vivant, photo)
                            prenom = e.nom.split()[0]
                            if type_ == "arrivee":
                                annoncer(f"Bonjour {prenom} ! Arrivée {a_la_minute(quand)}", VERT)
                            else:
                                annoncer(f"Au revoir {prenom} ! Départ {a_la_minute(quand)}", VERT)
            valid.nettoyer({v.piste for v in visages} | set(suivi.pistes))

            if args.sans_fenetre:
                continue
            messages[:] = [m for m in messages if m[2] > time.time()]
            image = dessiner_borne(image, visages, couleurs, messages, args.nom_borne, ecrivain)
            cv2.imshow(fenetre, image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.fermer()
        cv2.destroyAllWindows()


# =========================================================================== commandes employés
def cmd_employe_ajouter(args):
    reg = Registre(args.base)
    if reg.employe(args.nom):
        sys.exit(f"« {args.nom} » existe déjà. Pour ajouter des photos : python pointage.py employe photos \"{args.nom}\"")
    if args.matricule and reg.employe(args.matricule):
        sys.exit(f"Le matricule {args.matricule} est déjà utilisé.")
    if not args.sans_visage:
        vz.cmd_enregistrer(argparse.Namespace(nom=args.nom, images=args.images, nombre=args.nombre,
                                              camera=args.camera))
    e = reg.ajouter_employe(args.nom, args.matricule, args.service, args.debut, args.fin)
    print(f"✔ Employé ajouté : {e.nom} ({e.matricule or 'sans matricule'}, {e.service or 'sans service'}, "
          f"{e.heure_debut}-{e.heure_fin})")


def cmd_employe_photos(args):
    reg = Registre(args.base)
    e = reg.employe(args.nom)
    if e is None:
        sys.exit(f"Employé inconnu : {args.nom}")
    vz.cmd_enregistrer(argparse.Namespace(nom=e.nom, images=args.images, nombre=args.nombre,
                                          camera=args.camera))


def cmd_employe_liste(args):
    reg = Registre(args.base)
    connus = vz.charger_visages()
    employes = reg.employes()
    if not employes:
        print("Aucun employé."); return
    print(f"{'Nom':<24} {'Matricule':<10} {'Service':<14} {'Horaires':<12} {'Visage':<10} État")
    for e in employes:
        visage = f"{len(connus[e.nom])} photos" if e.nom in connus else "AUCUN"
        print(f"{e.nom:<24} {e.matricule or '':<10} {e.service:<14} {e.heure_debut}-{e.heure_fin:<6} "
              f"{visage:<10} {'actif' if e.actif else 'désactivé'}")


def cmd_employe_modifier(args):
    reg = Registre(args.base)
    try:
        e = reg.modifier_employe(args.nom, matricule=args.matricule, service=args.service,
                                 heure_debut=args.debut, heure_fin=args.fin)
    except (ValueError, sqlite3.IntegrityError) as err:
        sys.exit(str(err))
    print(f"✔ {e.nom} : {e.matricule or '-'}, {e.service or '-'}, {e.heure_debut}-{e.heure_fin}")


def cmd_employe_actif(args, actif: bool):
    reg = Registre(args.base)
    try:
        e = reg.modifier_employe(args.nom, actif=1 if actif else 0)
    except ValueError as err:
        sys.exit(str(err))
    print(f"✔ {e.nom} {'réactivé' if actif else 'désactivé'}.")
    if not actif and args.effacer_visage:
        f = vz.VISAGES / f"{e.nom}.npy"
        if f.exists():
            f.unlink()
        print("  Empreinte du visage effacée (l'historique des pointages est conservé).")


# =========================================================================== commandes suivi
def cmd_aujourdhui(args):
    reg = Registre(args.base)
    jour = args.date or date.today()
    employes = reg.employes(actifs=True)
    if not employes:
        print("Aucun employé actif."); return
    print(f"Présences du {jour:%d/%m/%Y}\n")
    print(f"{'Nom':<24} {'Service':<14} {'Arrivée':<8} {'Départ':<8} Statut")
    compte = Counter()
    for e in employes:
        j = reg.journee(e, jour, args.tolerance)
        if not j.passages:
            statut = "ABSENT" if jour.isoweekday() in args.jours_ouvres else "repos"
            compte["absents" if statut == "ABSENT" else "repos"] += 1
        else:
            statut = f"en retard de {duree_texte(j.retard)}" if j.retard else "à l'heure"
            if j.depart:
                statut += " — parti"
            compte["retards" if j.retard else "a_l_heure"] += 1
        print(f"{e.nom:<24} {e.service:<14} {a_la_minute(j.arrivee) if j.arrivee else '':<8} "
              f"{a_la_minute(j.depart) if j.depart else '':<8} {statut}")
    print(f"\nÀ l'heure : {compte['a_l_heure']}  |  En retard : {compte['retards']}  |  "
          f"Absents : {compte['absents']}")


def cmd_rapport(args):
    reg = Registre(args.base)
    if args.du or args.au:
        du, au = args.du or date.today().replace(day=1), args.au or date.today()
    else:
        annee, mois = args.mois
        du = date(annee, mois, 1)
        au = (date(annee + mois // 12, mois % 12 + 1, 1) - timedelta(days=1))
    lignes, resume = reg.rapport(du, au, args.jours_ouvres, args.tolerance)
    print(f"Rapport du {du:%d/%m/%Y} au {au:%d/%m/%Y}\n")
    print(f"{'Nom':<24} {'Jours ouvrés':>12} {'Présent':>8} {'Absences':>9} {'Retards':>8} "
          f"{'Retard total':>13} {'Heures':>8}")
    for e, s, minutes in resume:
        if not s["ouvres"] and not s["presents"]:
            continue
        print(f"{e.nom:<24} {s['ouvres']:>12} {s['presents']:>8} {s['absences']:>9} {s['retards']:>8} "
              f"{duree_texte(s['minutes_retard']):>13} {duree_texte(minutes):>8}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig : accents OK dans Excel
            w = csv.writer(f, delimiter=";")
            w.writerow(["Date", "Matricule", "Nom", "Service", "Arrivée", "Départ", "Durée",
                        "Retard (min)", "Départ anticipé (min)", "Statut"])
            for e, j in lignes:
                w.writerow([j.jour.strftime("%d/%m/%Y"), e.matricule or "", e.nom, e.service,
                            a_la_minute(j.arrivee) if j.arrivee else "",
                            a_la_minute(j.depart) if j.depart else "",
                            duree_texte(j.duree), j.retard or "", j.depart_anticipe or "", j.statut])
        print(f"\nDétail jour par jour : {args.csv}")


def cmd_historique(args):
    reg = Registre(args.base)
    e = reg.employe(args.nom)
    if e is None:
        sys.exit(f"Employé inconnu : {args.nom}")
    au = date.today()
    du = au - timedelta(days=args.jours)
    print(f"Pointages de {e.nom} depuis le {du:%d/%m/%Y}\n")
    print(f"{'N°':>5}  {'Date et heure':<17} {'Borne':<12} {'Ressembl.':>9} {'Vivant':>7}  Remarque")
    for p in reg.historique(e, du, au):
        remarque = []
        if p["manuel"]:
            remarque.append(f"ajout manuel : {p['motif']}")
        if p["annule"]:
            remarque.append(f"ANNULÉ : {p['motif_annulation']}")
        quand = datetime.fromisoformat(p["horodatage"])
        ressemblance = "" if p["ressemblance"] is None else f"{p['ressemblance']:.2f}"
        vivant = "" if p["vivant"] is None else f"{p['vivant']:.2f}"
        print(f"{p['id']:>5}  {quand:%d/%m/%Y %H:%M}  {p['borne'] or '':<12} "
              f"{ressemblance:>9} {vivant:>7}  {' ; '.join(remarque)}")


def cmd_corriger(args):
    reg = Registre(args.base)
    e = reg.employe(args.nom)
    if e is None:
        sys.exit(f"Employé inconnu : {args.nom}")
    pid = reg.pointer(e, args.le, borne="manuel", motif=args.motif)
    print(f"✔ Pointage n°{pid} ajouté pour {e.nom} le {args.le:%d/%m/%Y à %H:%M} ({args.motif}).")


def cmd_annuler(args):
    reg = Registre(args.base)
    if reg.annuler(args.id, args.motif):
        print(f"✔ Pointage n°{args.id} annulé ({args.motif}). Il reste visible dans l'historique.")
    else:
        sys.exit(f"Pointage n°{args.id} introuvable ou déjà annulé.")


def cmd_alertes(args):
    reg = Registre(args.base)
    lignes = reg.alertes(date.today() - timedelta(days=args.jours))
    if not lignes:
        print(f"Aucune alerte sur les {args.jours} derniers jours."); return
    for l in lignes:
        quand = datetime.fromisoformat(l["horodatage"])
        detail = f" (se fait passer pour {l['nom_suppose']}, score {l['score']:.2f})" if l["nom_suppose"] else ""
        print(f"{quand:%d/%m/%Y %H:%M}  {l['borne']:<10} {l['type']}{detail}  {l['photo'] or ''}")


# =========================================================================== analyseur d'arguments
def date_arg(texte: str) -> date:
    try:
        return datetime.strptime(texte, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError("date invalide (format 2026-11-07)") from None


def datetime_arg(texte: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(texte, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError("date et heure invalides (format \"2026-11-07 08:05\")")


def mois_arg(texte: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", texte)
    if not m or not 1 <= int(m.group(2)) <= 12:
        raise argparse.ArgumentTypeError("mois invalide (format 2026-11)")
    return int(m.group(1)), int(m.group(2))


def construire_parseur() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pointage des employés par reconnaissance faciale")
    p.add_argument("--base", default=str(BASE), help="fichier de la base (pointage.db)")
    sp = p.add_subparsers(dest="commande", required=True)

    # --- employés
    pe = sp.add_parser("employe", help="gérer les employés").add_subparsers(dest="action", required=True)
    a = pe.add_parser("ajouter", help="ajouter un employé et enregistrer son visage")
    a.add_argument("nom", type=vz.nom_valide)
    a.add_argument("--matricule")
    a.add_argument("--service", default="")
    a.add_argument("--debut", type=heure_valide, default="08:00", help="heure d'arrivée prévue (08:00)")
    a.add_argument("--fin", type=heure_valide, default="17:00", help="heure de départ prévue (17:00)")
    a.add_argument("--images", nargs="+", help="photos de l'employé (sinon : webcam)")
    a.add_argument("--nombre", type=int, default=15, help="captures webcam (15)")
    a.add_argument("--camera", type=int, default=0)
    a.add_argument("--sans-visage", action="store_true", help="créer la fiche sans enregistrer le visage")
    a.set_defaults(func=cmd_employe_ajouter)

    ph = pe.add_parser("photos", help="ajouter des captures du visage d'un employé")
    ph.add_argument("nom")
    ph.add_argument("--images", nargs="+")
    ph.add_argument("--nombre", type=int, default=15)
    ph.add_argument("--camera", type=int, default=0)
    ph.set_defaults(func=cmd_employe_photos)

    pe.add_parser("liste", help="liste des employés").set_defaults(func=cmd_employe_liste)

    m = pe.add_parser("modifier", help="modifier matricule, service ou horaires")
    m.add_argument("nom")
    m.add_argument("--matricule")
    m.add_argument("--service")
    m.add_argument("--debut", type=heure_valide)
    m.add_argument("--fin", type=heure_valide)
    m.set_defaults(func=cmd_employe_modifier)

    d = pe.add_parser("desactiver", help="désactiver un employé qui est parti")
    d.add_argument("nom")
    d.add_argument("--effacer-visage", action="store_true", help="effacer aussi son empreinte faciale")
    d.set_defaults(func=lambda args: cmd_employe_actif(args, False))
    r = pe.add_parser("reactiver", help="réactiver un employé")
    r.add_argument("nom")
    r.set_defaults(func=lambda args: cmd_employe_actif(args, True), effacer_visage=False)

    # --- borne
    b = sp.add_parser("borne", help="lancer une borne de pointage")
    b.add_argument("--source", default="0",
                   help="0 = webcam, 1 = 2e webcam, rtsp://… = caméra IP, ou un fichier vidéo")
    b.add_argument("--nom-borne", default="Entree", help="nom de la borne dans les rapports")
    b.add_argument("--seuil", type=float, default=SEUIL_POINTAGE, help=f"seuil de reconnaissance ({SEUIL_POINTAGE})")
    b.add_argument("--seuil-vivant", type=float, default=SEUIL_VIVANT,
                   help=f"seuil anti-fraude ({SEUIL_VIVANT}) : plus haut = plus strict")
    b.add_argument("--sans-antifraude", action="store_true", help="déconseillé")
    b.add_argument("--precis", action="store_true", help="petits visages lointains (caméras en hauteur)")
    b.add_argument("--largeur", type=int, default=1280, help="réduit l'image à cette largeur (1280)")
    b.add_argument("--photos-inconnus", action="store_true", help="garder aussi une photo des inconnus")
    b.add_argument("--plein-ecran", action="store_true")
    b.add_argument("--sans-fenetre", action="store_true", help="sans affichage (serveur)")
    b.set_defaults(func=cmd_borne)

    # --- suivi
    au = sp.add_parser("aujourdhui", help="présences du jour")
    au.add_argument("--date", type=date_arg, help="autre jour (2026-11-07)")
    au.add_argument("--tolerance", type=int, default=TOLERANCE_RETARD, help="minutes de tolérance (5)")
    au.add_argument("--jours-ouvres", type=jours_ouvres, default=jours_ouvres("1-5"),
                    help="jours travaillés, 1 = lundi (1-5)")
    au.set_defaults(func=cmd_aujourdhui)

    ra = sp.add_parser("rapport", help="rapport de présence (mois ou période)")
    ra.add_argument("--mois", type=mois_arg, default=(date.today().year, date.today().month),
                    help="mois (2026-11), par défaut le mois en cours")
    ra.add_argument("--du", type=date_arg)
    ra.add_argument("--au", type=date_arg)
    ra.add_argument("--csv", help="exporter le détail jour par jour (lisible par Excel)")
    ra.add_argument("--tolerance", type=int, default=TOLERANCE_RETARD)
    ra.add_argument("--jours-ouvres", type=jours_ouvres, default=jours_ouvres("1-5"))
    ra.set_defaults(func=cmd_rapport)

    hi = sp.add_parser("historique", help="pointages d'un employé")
    hi.add_argument("nom")
    hi.add_argument("--jours", type=int, default=31)
    hi.set_defaults(func=cmd_historique)

    co = sp.add_parser("corriger", help="ajouter un pointage oublié (avec motif)")
    co.add_argument("nom")
    co.add_argument("--le", type=datetime_arg, required=True, help='"2026-11-07 08:05"')
    co.add_argument("--motif", required=True)
    co.set_defaults(func=cmd_corriger)

    an = sp.add_parser("annuler", help="annuler un pointage (reste visible dans l'historique)")
    an.add_argument("id", type=int)
    an.add_argument("--motif", required=True)
    an.set_defaults(func=cmd_annuler)

    al = sp.add_parser("alertes", help="tentatives de fraude et visages inconnus")
    al.add_argument("--jours", type=int, default=7)
    al.set_defaults(func=cmd_alertes)
    return p


def main(argv=None):
    args = construire_parseur().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
