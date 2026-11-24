import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# Chemin vers le dossier principal du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Charger les clés depuis le fichier .env
load_dotenv(PROJECT_ROOT / ".env")

API_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("RAG_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "RAG_API_KEY manquante dans le fichier .env"
    )


def test_health():
    reponse = requests.get(
        f"{API_URL}/health",
        timeout=10
    )

    assert reponse.status_code == 200
    assert reponse.json()["status"] == "ok"

    print("GET /health : OK")


def test_question():
    reponse = requests.post(
        f"{API_URL}/ask",
        json={
            "question": "Quels concerts sont prévus à Paris ?"
        },
        headers={"X-API-Key": API_KEY},
        timeout=180
    )

    assert reponse.status_code == 200, reponse.text

    donnees = reponse.json()

    assert isinstance(donnees["reponse"], str)
    assert len(donnees["reponse"]) > 0
    assert isinstance(donnees["sources"], list)

    print("POST /ask : OK")
    print("Nombre de sources :", len(donnees["sources"]))


def test_question_vide():
    reponse = requests.post(
        f"{API_URL}/ask",
        json={"question": "   "},
        headers={"X-API-Key": API_KEY},
        timeout=10
    )

    assert reponse.status_code == 422

    print("Question vide : refusée (422)")


def test_sans_cle():
    reponse = requests.post(
        f"{API_URL}/ask",
        json={"question": "Bonjour"},
        timeout=10
    )

    assert reponse.status_code == 401

    print("Accès sans clé : refusé (401)")


def test_rebuild_protege():
    reponse = requests.post(
        f"{API_URL}/rebuild",
        headers={"X-API-Key": API_KEY},
        timeout=10
    )

    assert reponse.status_code == 403

    print("Rebuild sans droits admin : refusé (403)")


if __name__ == "__main__":

    test_health()
    test_question_vide()
    test_sans_cle()
    test_rebuild_protege()

    print("\nTous les tests de sécurité sont réussis.")