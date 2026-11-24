
import os
import sys
import secrets
import subprocess

from pathlib import Path
from threading import Lock
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from src.rag_service import RAGService


# Chemins
BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("RAG_API_KEY")
ADMIN_KEY = os.getenv("RAG_ADMIN_KEY")

if not API_KEY or not ADMIN_KEY:
    raise RuntimeError(
        "RAG_API_KEY et RAG_ADMIN_KEY sont obligatoires "
        "dans le fichier .env."
    )

if API_KEY == ADMIN_KEY:
    raise RuntimeError(
        "Les clés utilisateur et administrateur "
        "doivent être différentes."
    )


# Sécurité : clé transmise dans l'en-tête HTTP
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verifier_utilisateur(
    cle: str | None = Depends(api_key_header)
):
    if cle is None or not (
        secrets.compare_digest(cle, API_KEY)
        or secrets.compare_digest(cle, ADMIN_KEY)
    ):
        raise HTTPException(
            status_code=401,
            detail="Clé API absente ou invalide."
        )


def verifier_admin(
    cle: str | None = Depends(api_key_header)
):
    if cle is None or not secrets.compare_digest(
        cle, ADMIN_KEY
    ):
        raise HTTPException(
            status_code=403,
            detail="Accès administrateur requis."
        )


# Modèles de requête et de réponse
class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Question sur les événements culturels"
    )

    @field_validator("question")
    @classmethod
    def question_non_vide(cls, valeur):
        valeur = valeur.strip()

        if not valeur:
            raise ValueError(
                "La question ne peut pas être vide."
            )

        return valeur


class SourceEvenement(BaseModel):
    titre: str | None = None
    lieu: str | None = None
    date_fin: str | None = None
    url: str | None = None


class QuestionResponse(BaseModel):
    reponse: str
    sources: list[SourceEvenement]


# Charger le RAG une seule fois au démarrage
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RAGService()
    app.state.rebuild_lock = Lock()

    yield


app = FastAPI(
    title="Pulls-Events - API RAG",
    description=(
        "API REST de recommandations culturelles "
        "avec FAISS, LangChain et Mistral."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# GET /health
@app.get(
    "/health",
    summary="Vérifier le fonctionnement de l'API"
)
def health():
    return {"status": "ok"}


# POST /ask
@app.post(
    "/ask",
    response_model=QuestionResponse,
    summary="Poser une question au chatbot RAG"
)
def ask(
    demande: QuestionRequest,
    request: Request,
    _=Depends(verifier_utilisateur)
):
    try:
        return request.app.state.rag.poser_question(
            demande.question
        )

    except Exception:
        # Ne pas exposer de détails internes au client
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de la génération de la réponse."
        )


# POST /rebuild
@app.post(
    "/rebuild",
    summary="Reconstruction de l'index FAISS",
    description=(
        "Réservé aux possesseur de la clé. "
        "Index reconstruit à partir des données clean."
    )
)
def rebuild(
    request: Request,
    _=Depends(verifier_admin)
):
    verrou = request.app.state.rebuild_lock

    if not verrou.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Une reconstruction est déjà en cours."
        )

    try:
        script = BASE_DIR / "src" / "build_faiss_index.py"

        # Exécuter le script existant avec le même Python
        resultat = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=1800,
            check=False
        )

        if resultat.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail="La reconstruction de l'index a échoué."
            )

        # Actualiser l'index utilisé par le chatbot
        nombre = request.app.state.rag.recharger_index()

        return {
            "message": "Index FAISS reconstruit et rechargé.",
            "vecteurs": nombre
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Délai de reconstruction dépassé."
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Erreur pendant la reconstruction."
        )

    finally:
        verrou.release()
