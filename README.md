# Facial_reconaissance

Reconnaissance faciale et détection des émotions en Python, en direct avec la webcam, sur une vidéo ou sur une photo.

Deux outils :

- **`pointage.py`** : pointage des employés par reconnaissance faciale (arrivées, départs, retards,
  absences, rapports Excel), avec anti-fraude contre les photos et les écrans.
- **`visage.py`** : reconnaissance des personnes et détection des émotions
  (neutre, joie, surprise, tristesse, colère, dégoût, peur, mépris).

## Fonctionnalités

- Détection de plusieurs visages à la fois (YuNet), y compris les petits visages au fond d'une photo
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

Les modèles (environ 75 Mo) se téléchargent tout seuls au premier lancement, dans `modeles/`.
Si le téléchargement échoue, le programme affiche le lien pour les récupérer à la main.

## Pointage des employés

### 1. Inscrire les employés

```bash
# avec la webcam : l'employé regarde la caméra et tourne doucement la tête (15 captures)
python pointage.py employe ajouter "Awa Diallo" --matricule E001 --service Compta --debut 08:00 --fin 17:00

# ou avec des photos (un seul visage par photo, de face, bien éclairé)
python pointage.py employe ajouter "Moussa Bah" --matricule E002 --images moussa1.jpg moussa2.jpg moussa3.jpg

python pointage.py employe liste
python pointage.py employe photos "Awa Diallo"          # ajouter des captures si elle est mal reconnue
python pointage.py employe modifier "Awa Diallo" --debut 07:30
python pointage.py employe desactiver "Moussa Bah" --effacer-visage   # départ de l'entreprise
```

### 2. Lancer la borne

```bash
python pointage.py borne                                  # webcam
python pointage.py borne --plein-ecran --nom-borne "Entrée principale"
python pointage.py borne --source "rtsp://admin:motdepasse@192.168.1.64:554/Streaming/Channels/101" --nom-borne Portail
```

L'employé se place devant la caméra ; en moins d'une seconde la borne affiche
« Bonjour Awa ! Arrivée 08:02 » ou « Au revoir Awa ! Départ 17:05 ».

- **Premier passage de la journée = arrivée, dernier passage = départ.**
- Un 2e passage dans les 2 minutes est ignoré (« déjà pointé »).
- La personne doit être reconnue sur 5 images de suite : une erreur sur une seule image ne pointe personne.
- **Anti-fraude** : une photo, un écran de téléphone ou une vidéo présentée devant la caméra est refusée,
  et une alerte est enregistrée avec la photo de la tentative.
- Une photo du visage est gardée pour chaque pointage (`photos_pointage/`), en cas de contestation.
- Caméra IP ou de surveillance : reconnexion automatique si le flux coupe, pas de retard qui s'accumule.
- Plusieurs bornes peuvent tourner en même temps sur la même base (une commande par caméra, `--nom-borne` différent).
- Un employé inscrit pendant que la borne tourne est reconnu sans la redémarrer (30 s maximum).

| Option de `borne` | Rôle |
|---|---|
| `--source 1` | 2e webcam, `rtsp://…` pour une caméra IP, ou un fichier vidéo |
| `--plein-ecran` | pour une tablette ou un écran à l'entrée |
| `--precis` | caméra placée loin ou en hauteur (petits visages), plus lent |
| `--seuil 0.45` | reconnaissance plus stricte |
| `--seuil-vivant 0.9` | anti-fraude plus stricte |
| `--photos-inconnus` | garder aussi la photo des personnes non reconnues |
| `--sans-fenetre` | sans écran (serveur) : les pointages s'affichent dans la console |

**Conseil d'installation** : une caméra **à l'entrée, à hauteur de visage**, bien éclairée, sans contre-jour.
Les caméras de surveillance au plafond filment de haut et de loin : la reconnaissance y est beaucoup moins fiable.

### 3. Suivre les présences

```bash
python pointage.py aujourdhui                             # qui est là, en retard, absent
python pointage.py rapport                                # mois en cours
python pointage.py rapport --mois 2026-11 --csv novembre.csv   # détail jour par jour pour Excel / la paie
python pointage.py rapport --du 2026-11-01 --au 2026-11-15 --jours-ouvres 1-6   # lundi à samedi
python pointage.py historique "Awa Diallo"
python pointage.py alertes                                # tentatives de fraude des 7 derniers jours
```

Le rapport donne pour chaque employé : jours présents, absences, nombre et total des retards
(au-delà de 5 minutes de tolérance, réglable avec `--tolerance`), et heures de présence.
Un employé n'est pas compté absent avant sa date d'inscription, ni le jour même avant la fin de la journée.

### 4. Corriger

```bash
python pointage.py corriger "Awa Diallo" --le "2026-11-07 08:05" --motif "panne de la caméra"
python pointage.py annuler 42 --motif "doublon"
```

Chaque correction garde son motif ; un pointage annulé reste visible dans l'historique (traçabilité).

### Cadre légal

Le visage est une **donnée biométrique**. Avant de mettre le pointage en service :

- vérifiez les obligations auprès de l'autorité de protection des données du pays
  (par exemple la CDP au Sénégal) : une autorisation préalable est souvent exigée ;
- informez les employés par écrit et proposez une autre solution (badge, code) à ceux qui refusent ;
- limitez la reconnaissance au point de pointage, sans surveillance continue des postes de travail ;
- effacez l'empreinte faciale des employés qui quittent l'entreprise (`desactiver --effacer-visage`).

La détection des émotions n'est **pas** utilisée par le pointage.

## Reconnaissance et émotions (`visage.py`)

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

# Analyser une photo (crée photo_analyse.jpg ; --rapide pour ignorer les petits visages)
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
| `--precis` | trouver aussi les petits visages lointains (plus lent) |
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

Les empreintes des visages (`visages_connus/`), la base de pointage (`pointage.db`) et les photos
(`photos_pointage/`) restent sur votre ordinateur et ne sont jamais envoyées ailleurs ni mises dans Git.
Demandez l'accord des personnes avant d'enregistrer leur visage.

## Modèles utilisés

- [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) et
  [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) — OpenCV Zoo
- [FER+](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) — ONNX Model Zoo
- [MiniFASNetV2](https://github.com/yakhyo/face-anti-spoofing) (anti-fraude, d'après Silent-Face-Anti-Spoofing de MiniVision)
