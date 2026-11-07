# Facial_reconaissance

Reconnaissance faciale et détection des émotions en Python, en direct avec la webcam, sur une vidéo ou sur une photo.

Le programme reconnaît les personnes enregistrées et affiche l'humeur de chaque visage :
**neutre, joie, surprise, tristesse, colère, dégoût, peur, mépris**, avec un pourcentage de confiance.

## Fonctionnalités

- Détection de plusieurs visages à la fois (YuNet)
- Reconnaissance des personnes enregistrées (SFace, similarité cosinus)
- 8 émotions (FER+), lissées dans le temps pour éviter que l'affichage clignote
- Suivi des visages d'une image à l'autre : le nom reste stable même si une image est ratée
- Enregistrement guidé : refuse les captures floues, trop petites ou trop semblables
- Webcam, fichier vidéo (avec export de la vidéo annotée) ou photo
- Journal CSV des humeurs et rapport par personne
- Fonctionne hors ligne une fois les modèles téléchargés, sans GPU

## Installation

Python 3.9 ou plus récent.

```bash
git clone https://github.com/elhadjiibrahima/Facial_reconaissance.git
cd Facial_reconaissance
pip install -r requirements.txt
```

Les 3 modèles (environ 40 Mo) se téléchargent tout seuls au premier lancement, dans `modeles/`.
Si le téléchargement échoue, le programme affiche le lien pour les récupérer à la main.

## Utilisation

```bash
# 1. Enregistrer une personne (webcam : tournez doucement la tête)
python visage.py enregistrer Ibrahima

#    ... ou à partir de photos (un seul visage par photo)
python visage.py enregistrer Awa --images awa1.jpg awa2.jpg awa3.jpg

# 2. Analyse en direct (q = quitter, s = capture d'écran)
python visage.py lancer

# Analyser une vidéo et enregistrer le résultat
python visage.py lancer --video reunion.mp4 --sortie reunion_analyse.mp4

# Noter les humeurs toutes les 5 secondes, puis afficher le résumé
python visage.py lancer --journal humeurs.csv
python visage.py rapport humeurs.csv

# Analyser une photo (crée photo_analyse.jpg)
python visage.py photo photo.jpg

# Gérer les personnes
python visage.py liste
python visage.py supprimer Awa
```

### Options utiles de `lancer`

| Option | Rôle |
|---|---|
| `--camera 1` | utiliser une autre webcam |
| `--sauter 1` | analyser une image sur deux (PC lent) |
| `--largeur 640` | réduire l'image pour aller plus vite |
| `--seuil 0.4` | reconnaissance plus stricte (moins d'erreurs, plus d'« Inconnu ») |
| `--sans-fenetre` | sans affichage, avec `--sortie` ou `--journal` |

## Conseils pour de bons résultats

- Enregistrez chaque personne avec 15 captures ou plus, sous des éclairages différents.
- Si quelqu'un est souvent « Inconnu », ajoutez-lui des captures : `enregistrer` complète les anciennes.
- Si deux personnes sont confondues, augmentez `--seuil`.
- L'humeur est une **estimation** à partir de l'expression du visage : elle marche mieux de face et bien éclairé.

## Tests

```bash
pip install pytest
python -m pytest
```

## Vie privée

Les empreintes des visages restent sur votre ordinateur (`visages_connus/`) et ne sont jamais envoyées ailleurs.
Demandez l'accord des personnes avant d'enregistrer leur visage.

## Modèles utilisés

- [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) et
  [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) — OpenCV Zoo
- [FER+](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) — ONNX Model Zoo
