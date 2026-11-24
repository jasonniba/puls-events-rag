
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = BASE_DIR / "data" / "faiss_index"


# Charger le même modèle que pour l'indexation
model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

# Charger l'index déjà construit
vectorstore = FAISS.load_local(
    str(INDEX_PATH),
    embeddings=model,
    allow_dangerous_deserialization=True
)


def rechercher_evenements(question, k=5):
    aujourd_hui = datetime.now(timezone.utc)

    # Récupérer davantage de chunks pour pouvoir
    # éliminer les événements terminés et les doublons
    resultats = vectorstore.similarity_search(
        question,
        k=100
    )

    evenements = []
    ids_vus = set()

    for doc in resultats:
        metadata = doc.metadata

        date_fin = pd.to_datetime(
            metadata.get("date_fin"),
            errors="coerce",
            utc=True
        )

        # Ignorer les dates invalides ou passées
        if pd.isna(date_fin) or date_fin < aujourd_hui:
            continue

        # Éviter les événements répétés
        identifiant = metadata.get("id")

        if identifiant in ids_vus:
            continue

        ids_vus.add(identifiant)

        evenements.append({
            "id": identifiant,
            "titre": metadata.get("titre"),
            "lieu": metadata.get("lieu"),
            "ville": metadata.get("ville"),
            "date_debut": metadata.get("date_debut"),
            "date_fin": metadata.get("date_fin"),
            "url": metadata.get("url"),
            "description": doc.page_content
        })

        if len(evenements) >= k:
            break

    return evenements


if __name__ == "__main__":

    question = "Quels concerts ou spectacles sont prévus à Paris ?"

    resultats = rechercher_evenements(question)

    for evenement in resultats:
        print("\nTitre :", evenement["titre"])
        print("Lieu :", evenement["lieu"])
        print("Date de fin :", evenement["date_fin"])
        print("URL :", evenement["url"])
        print("-" * 50)
