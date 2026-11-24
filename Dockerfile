
FROM python:3.12-slim

# Configuration Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/opt/huggingface

WORKDIR /app

# PyTorch CPU uniquement : aucune carte graphique nécessaire
RUN python -m pip install --no-cache-dir --timeout 120 \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Dépendances FastAPI et RAG avec Mistral API
RUN python -m pip install --no-cache-dir --timeout 120 \
    fastapi \
    "uvicorn[standard]" \
    python-dotenv \
    requests \
    pandas \
    numpy \
    langchain-core \
    langchain-community \
    langchain-huggingface \
    langchain-text-splitters \
    langchain-mistralai \
    "transformers==4.46.3" \
    "sentence-transformers==3.3.1" \
    faiss-cpu

# Télécharger le modèle d'embeddings pendant le build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Copier la version actuelle du code
COPY docs-rag-agent/src /app/src

# Dossier destiné aux données FAISS montées depuis Windows
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
