
import requests
import pandas as pd
from pathlib import Path

# Chemins du projet
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# API OpenDataSoft - données OpenAgenda
url = (
    "https://public.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/evenements-publics-openagenda/exports/csv"
)

# Paramètrage du projet
ville = "Paris"
date_limite = (
    pd.Timestamp.today().normalize()
    - pd.DateOffset(years=1)
).strftime("%Y-%m-%d")

# transmission des Filtres directement appliqués par l'API
params = {
    "where": (
        f"location_city = '{ville}' "
        f"AND lastdate_end >= '{date_limite}'"
    ),
    "delimiter": ";"
}

# Téléchargement des données
print("Récupération des événements OpenAgenda...")

response = requests.get(
    url,
    params=params,
    timeout=120
)

response.raise_for_status()

# Sauvegarde du CSV brut
fichier = DATA_DIR / "openagenda_api.csv"
fichier.write_bytes(response.content)

# Vérification avec Pandas
df_api = pd.read_csv(fichier, sep=";", low_memory=False)

print("Téléchargement terminé !")
print("Dimensions :", df_api.shape)
print("Fichier :", fichier)

print("\nPremières lignes :")
print(df_api.head())
