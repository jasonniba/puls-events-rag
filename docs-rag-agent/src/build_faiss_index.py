import pandas as pd
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Chemins du projet
BASE_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = BASE_DIR / "data" / "processed" / "openagenda_clean.csv"
INDEX_PATH = BASE_DIR / "data" / "faiss_index"


# 2. Chargement des données
df = pd.read_csv(CSV_PATH)

print("Événements chargés :", len(df))

df = df.dropna(subset=["Identifiant", "texte_rag"])
df = df[df["texte_rag"].str.strip() != ""].copy()

# Vérifier que les identifiants sont uniques
if df["Identifiant"].duplicated().any():
    raise ValueError("Identifiants d'événements dupliqués")


# 3. Création des documents LangChain
documents = []

for _, row in df.iterrows():

    metadata = {
        "id": str(row["Identifiant"]),
        "titre": str(row["Titre"]),
        "ville": str(row["Ville"]),
        "lieu": str(row["Nom du lieu"]),
        "date_debut": str(row["Première date - Début"]),
        "date_fin": str(row["Dernière date - Fin"]),
        "url": str(row["URL canonique"])
    }

    document = Document(
        page_content=str(row["texte_rag"]),
        metadata=metadata
    )

    documents.append(document)

print("Documents créés :", len(documents))


# 4. Découpage en chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Chunks créés :", len(chunks))

# Vérifier que tous les événements sont représentés
ids_indexes = {chunk.metadata["id"] for chunk in chunks}

assert len(ids_indexes) == len(df), (
    "Certains événements ne sont pas représentés"
)


# 5. Modèle d'embeddings local
model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 32}
)


# 6. Construction de l'index FAISS
print("Construction de l'index FAISS...")

batch_size = 500

vectorstore = FAISS.from_documents(
    documents=chunks[:batch_size],
    embedding=model
)

for i in range(batch_size, len(chunks), batch_size):
    lot = chunks[i:i + batch_size]

    vectorstore.add_documents(lot)

    print(
        f"{min(i + batch_size, len(chunks))}"
        f"/{len(chunks)} chunks indexés"
    )


# 7. Vérification
assert vectorstore.index.ntotal == len(chunks), (
    "Le nombre de vecteurs ne correspond pas aux chunks"
)

print("Vecteurs indexés :", vectorstore.index.ntotal)
print("Dimensions :", vectorstore.index.d)


# 8. Sauvegarde
vectorstore.save_local(str(INDEX_PATH))

print("Index FAISS sauvegardé :", INDEX_PATH)
