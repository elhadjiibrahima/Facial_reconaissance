"""Configuration centralisée du projet Facial_reconaissance."""
from pathlib import Path

# Seuils
SEUIL_RECONNAISSANCE = 0.363
SEUIL_POINTAGE = 0.40
SEUIL_VIVANT = 0.80
IMAGES_CONFIRMATION = 5
DELAI_DOUBLON = 120          # secondes
TOLERANCE_RETARD = 5         # minutes

# Chemins
DOSSIER = Path(__file__).resolve().parent
MODELES = DOSSIER / "modeles"
VISAGES = DOSSIER / "visages_connus"
BASE_POINTAGE = DOSSIER / "pointage.db"
PHOTOS_POINTAGE = DOSSIER / "photos_pointage"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = DOSSIER / "facial.log"
