# Facial_reconaissance

[![Tests](https://github.com/elhadjiibrahima/Facial_reconaissance/actions/workflows/tests.yml/badge.svg)](https://github.com/elhadjiibrahima/Facial_reconaissance/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reconnaissance faciale et détection des émotions en Python, en direct avec la webcam, sur une vidéo ou sur une photo.

Deux outils :

- **`pointage.py`** : pointage des employés par reconnaissance faciale (arrivées, départs, retards, absences, rapports Excel), avec anti-fraude contre les photos et les écrans.
- **`visage.py`** : reconnaissance des personnes et détection des émotions (neutre, joie, surprise, tristesse, colère, dégoût, peur, mépris).

## Nouveautés

- **Logging structuré** (`logging_config.py` + fichier `facial.log`)
- **Configuration centralisée** (`config.py`)
- **Dashboard web** des présences (stdlib uniquement) : `python dashboard.py`

## Fonctionnalités

- Détection de plusieurs visages à la fois (YuNet), y compris les petits visages au fond d'une photo
- Reconnaissance des personnes enregistrées (SFace, similarité cosinus)
- 8 émotions (FER+), lissées dans le temps pour éviter que l'affichage clignote
- Suivi des visages d'une image à l'autre : le nom reste stable même si une image est ratée
- Enregistrement guidé : refuse les captures floues, trop petites ou trop semblables
- Webcam, fichier vidéo (avec export de la vidéo annotée) ou photo
- Journal CSV des humeurs et rapport par personne
- Fonctionne hors ligne une fois les modèles téléchargés, sans GPU
- **Anti-fraude** (MiniFASNet) contre les photos et écrans
- Base SQLite avec traçabilité complète des pointages et corrections
- Dashboard web léger pour le suivi des présences

## Installation

Python 3.9 ou plus récent.

```bash
git clone https://github.com/elhadjiibrahima/Facial_reconaissance.git
cd Facial_reconaissance

# Environnement virtuel recommandé
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

Les modèles (environ 75 Mo) se téléchargent tout seuls au premier lancement, dans `modeles/`.

## Dashboard web

```bash
python dashboard.py
```

Puis ouvrir http://127.0.0.1:8080 dans le navigateur.  
Il affiche les présences du jour et le résumé du mois, et se rafraîchit toutes les 30 secondes.

## Pointage des employés

### 1. Inscrire les employés

```bash
python pointage.py employe ajouter "Awa Diallo" --matricule E001 --service Compta --debut 08:00 --fin 17:00
python pointage.py employe ajouter "Moussa Bah" --matricule E002 --images moussa1.jpg moussa2.jpg
python pointage.py employe liste
```

### 2. Lancer la borne

```bash
python pointage.py borne
python pointage.py borne --plein-ecran --nom-borne "Entrée principale"
```

### 3. Suivre les présences

```bash
python pointage.py aujourdhui
python pointage.py rapport --mois 2026-11 --csv novembre.csv
python pointage.py historique "Awa Diallo"
python pointage.py alertes
```

### 4. Corriger

```bash
python pointage.py corriger "Awa Diallo" --le "2026-11-07 08:05" --motif "panne de la caméra"
python pointage.py annuler 42 --motif "doublon"
```

## Reconnaissance et émotions (`visage.py`)

```bash
python visage.py enregistrer Ibrahima
python visage.py lancer
python visage.py lancer --video reunion.mp4 --sortie reunion_analyse.mp4
python visage.py photo photo.jpg
```

## Tests

```bash
python -m pytest
```

## Vie privée

Les empreintes des visages, la base de pointage et les photos restent uniquement sur votre machine.

## Licence

MIT — voir le fichier [LICENSE](LICENSE).
