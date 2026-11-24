# Projet 9 – Puls-Events : Système de recommandation RAG

## Présentation du projet

Ce projet a été développé dans le cadre de la formation Data Scientist d'OpenClassrooms.

**Puls-Events** est un prototype de chatbot intelligent permettant de recommander des événements culturels à Paris à partir des données OpenAgenda.

Le système repose sur une architecture **RAG (Retrieval-Augmented Generation)**, qui associe une recherche documentaire vectorielle à un modèle de langage.

L'utilisateur peut poser une question en langage naturel, par exemple :

> Quels concerts sont prévus à Paris ?

Le chatbot recherche les événements pertinents dans FAISS, puis utilise l'API Mistral pour générer une réponse contextualisée accompagnée des sources.

## 1. Objectifs

- Collecter et préparer les données OpenAgenda.
- Transformer les descriptions des événements en vecteurs.
- Construire un index FAISS permettant la recherche sémantique.
- Développer un chatbot RAG avec LangChain et Mistral.
- Exposer le système à travers une API REST FastAPI.
- Conteneuriser l'API avec Docker.
- Proposer une interface Streamlit pour la démonstration.

## 2. Technologies utilisées

| Technologie | Rôle |
|---|---|
| Python 3.12 | Développement |
| Pandas | Nettoyage et transformation |
| Hugging Face | Génération des embeddings |
| FAISS | Indexation et recherche vectorielle |
| LangChain | Orchestration RAG |
| Mistral API | Génération des réponses |
| FastAPI | API REST |
| Streamlit | Interface utilisateur |
| Docker | Conteneurisation |
| Docker Compose | Exécution du conteneur |
| Pytest | Tests automatisés |

**Modèle d'embeddings :** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

**Dimensions des vecteurs :** 384

**Modèle de génération :** `mistral-small-latest`

Les embeddings sont calculés sur CPU. Le modèle Mistral est appelé via une API distante et n'est pas exécuté localement.

## 3. Données OpenAgenda

Les données proviennent du jeu de données OpenAgenda accessible via l'API OpenDataSoft.

Le traitement comprend :

1. Récupération des événements.
2. Sélection des événements concernant Paris.
3. Nettoyage des champs textuels.
4. Préparation des dates et des informations de localisation.
5. Construction du texte utilisé pour la recherche sémantique.

### Résultats obtenus

| Indicateur | Résultat |
|---|---:|
| Événements récupérés | 11 756 |
| Événements nettoyés | 11 718 |
| Chunks créés | 38 171 |
| Dimensions des embeddings | 384 |
| Taille de l'index | 38 171 vecteurs |

Fichiers principaux :

- `data/raw/openagenda_api.csv` : données brutes.
- `data/processed/openagenda_clean.csv` : données nettoyées.
- `data/faiss_index/` : index vectoriel et métadonnées.

## 4. Architecture du système RAG

Le traitement d'une question suit les étapes suivantes :

```text
Utilisateur
    |
    v
Streamlit
    |
    | Requête HTTP POST /ask
    v
FastAPI (conteneur Docker)
    |
    v
RAGService / LangChain
    |
    +--> Hugging Face Embeddings (CPU)
    |         |
    |         v
    |    Recherche FAISS
    |         |
    |         v
    |    Événements pertinents
    |
    v
Construction du prompt enrichi
    |
    v
Mistral API (LLM distant)
    |
    v
Réponse + sources OpenAgenda
    |
    v
Streamlit
```

### Fonctionnement

La question utilisateur est vectorisée avec le même modèle Hugging Face que celui utilisé pour construire l'index.

FAISS recherche les descriptions les plus proches sémantiquement.

Un filtrage permet d'exclure les événements terminés et d'éviter les doublons.

Jusqu'à cinq événements sont sélectionnés et transmis à Mistral avec leurs informations.

Le modèle génère ensuite une réponse en français en s'appuyant sur ces événements.

## 5. Structure du projet

```text
Projet9_systeme_rag/
|
|-- docs-rag-agent/
|   |-- data/
|   |   |-- raw/
|   |   |-- processed/
|   |   |-- faiss_index/
|   |
|   |-- src/
|   |   |-- fetch_openagenda.py
|   |   |-- preprocess_openagenda.py
|   |   |-- build_faiss_index.py
|   |   |-- search_events.py
|   |   |-- rag_service.py
|   |   |-- api.py
|   |
|   |-- tests/
|   |-- streamlit_app.py
|   |-- requirements.txt
|   |-- embeddings.ipynb
|   |-- indexation_faiss.ipynb
|   |-- preprocessing_openagenda.ipynb
|
|-- Dockerfile
|-- docker-compose.yml
|-- .dockerignore
|-- .gitignore
|-- pyproject.toml
|-- README.md
```

### Rôle des principaux fichiers

| Fichier | Fonction |
|---|---|
| `fetch_openagenda.py` | Récupération des événements |
| `preprocess_openagenda.py` | Prétraitement des données |
| `build_faiss_index.py` | Construction de l'index vectoriel |
| `search_events.py` | Recherche sémantique |
| `rag_service.py` | Logique RAG : FAISS, LangChain et Mistral |
| `api.py` | Endpoints FastAPI |
| `streamlit_app.py` | Interface de démonstration |
| `Dockerfile` | Construction de l'image Docker |
| `docker-compose.yml` | Configuration du conteneur |

## 6. Prérequis

- Python 3.12
- Docker Desktop
- Git
- Connexion Internet pour Mistral API
- Clé Mistral Studio avec un quota de génération disponible

Une carte graphique n'est pas nécessaire.

Le LLM est exécuté à distance via l'API Mistral. Les embeddings Hugging Face et FAISS fonctionnent sur CPU.

## 7. Installation locale

Depuis la racine du projet, sur Windows :

### Créer l'environnement virtuel

```powershell
py -3.12 -m venv .venv
```

### Activer l'environnement

```powershell
.\.venv\Scripts\Activate.ps1
```

Avec Git Bash :

```bash
source .venv/Scripts/activate
```

### Installer les dépendances

Pour installer PyTorch CPU :

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Puis installer le projet et les outils de développement :

```powershell
python -m pip install -e ".[dev]"
```

Les dépendances principales sont déclarées dans `pyproject.toml`.

## 8. Configuration des variables d'environnement

Créer un fichier `.env` à la racine du projet :

```dotenv
MISTRAL_API_KEY=VOTRE_CLE_MISTRAL
MISTRAL_MODEL=mistral-small-latest

RAG_API_KEY=VOTRE_CLE_UTILISATEUR
RAG_ADMIN_KEY=VOTRE_CLE_ADMIN

RAG_API_URL=http://127.0.0.1:8001
STREAMLIT_DEMO_MODE=false
```

Créer également `.env.docker` :

```dotenv
MISTRAL_API_KEY=VOTRE_CLE_MISTRAL
MISTRAL_MODEL=mistral-small-latest

RAG_API_KEY=VOTRE_CLE_UTILISATEUR
RAG_ADMIN_KEY=VOTRE_CLE_ADMIN
```

Les clés `RAG_API_KEY` et `RAG_ADMIN_KEY` doivent être différentes.

Les clés doivent correspondre entre `.env` et `.env.docker` pour permettre à Streamlit de communiquer avec l'API.

Ces fichiers contiennent des secrets et ne doivent jamais être publiés sur GitHub.

## 9. Construction de l'index FAISS

Le projet utilise un index FAISS déjà construit à partir des événements OpenAgenda nettoyés.

Les fichiers nécessaires sont :

```text
docs-rag-agent/data/faiss_index/
|-- index.faiss
|-- index.pkl
```

Si l'index n'est pas fourni, il faut disposer du CSV nettoyé et lancer le script depuis `docs-rag-agent` :

```bash
cd docs-rag-agent
python src/build_faiss_index.py
cd ..
```

Cette opération peut prendre du temps, car elle calcule les embeddings des descriptions.

Le script de reconstruction utilise le modèle Hugging Face et ne nécessite pas d'appel au LLM Mistral.

## 10. Exécution de l'API avec Docker

Docker Desktop doit être démarré.

Depuis la racine du projet :

### Construire l'image Docker

```bash
docker build -t puls-events-rag:latest .
```

### Démarrer le conteneur

```bash
docker compose up -d
```

### Vérifier les conteneurs

```bash
docker compose ps
```

### Consulter les journaux

```bash
docker compose logs --tail=50
```

### Arrêter le conteneur

```bash
docker compose down
```

Le port de l'API est exposé localement :

`http://127.0.0.1:8001`

La documentation Swagger est accessible à :

`http://127.0.0.1:8001/docs`

Le dossier `docs-rag-agent/data` est monté dans Docker sous `/app/data`.

L'index FAISS reste stocké localement.

## 11. Endpoints FastAPI

| Méthode | Endpoint | Rôle |
|---|---|---|
| GET | `/health` | Vérifier que l'API répond |
| POST | `/ask` | Interroger le chatbot RAG |
| POST | `/rebuild` | Reconstruire et recharger l'index FAISS |

### Vérifier l'API

```bash
curl http://127.0.0.1:8001/health
```

Résultat attendu :

```json
{"status":"ok"}
```

### Interroger le chatbot

Dans Swagger, utiliser `POST /ask` avec la clé `RAG_API_KEY`.

Exemple :

```json
{
  "question": "Quels concerts sont prévus à Paris ?"
}
```

La réponse contient une réponse générée par Mistral et une liste de sources OpenAgenda.

L'endpoint `/rebuild` est réservé aux administrateurs. Il nécessite `RAG_ADMIN_KEY`.

Il permet de reconstruire l'index à partir du CSV nettoyé, puis de le recharger dans l'API. Il ne télécharge pas automatiquement les nouvelles données OpenAgenda.

## 12. Démonstration avec Streamlit

L'interface Streamlit est lancée séparément du conteneur Docker.

Depuis la racine du projet, avec l'environnement virtuel activé :

```bash
python -m streamlit run docs-rag-agent/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Ouvrir dans le navigateur :

`http://127.0.0.1:8501`

L'interface permet :

- De poser une question en français.
- D'obtenir des recommandations culturelles.
- De consulter les événements sources.
- D'accéder aux liens OpenAgenda.

Pour les réponses réelles, utiliser :

```dotenv
STREAMLIT_DEMO_MODE=false
```

Le mode démonstration simulé peut être utilisé pour tester l'interface, mais ne constitue pas une validation du RAG.

### Scénarios de démonstration

1. Quels concerts sont prévus à Paris ?
2. Je cherche une exposition artistique à Paris.
3. Quelles activités culturelles peut-on faire avec des enfants à Paris ?

Les questions sont traitées indépendamment. Le POC ne conserve pas l'historique de conversation.

## 13. Tests et résultats

Les vérifications réalisées comprennent :

- Construction d'un index FAISS de 38 171 vecteurs.
- Validation de la recherche sémantique.
- Recherche de cinq événements distincts.
- Temps de recherche FAISS avec embeddings CPU dans Docker : **0,805 seconde**.
- Validation de l'endpoint `/health` : HTTP 200.
- Validation de `/ask` avec Mistral API.
- Affichage de recommandations réelles dans Streamlit.
- Première évaluation manuelle d'une réponse du chatbot, classée **Correcte**.

La mesure de 0,805 seconde concerne uniquement la recherche sémantique avec embeddings CPU, et non le temps total de génération de la réponse.

Aucun score global de similarité des réponses ou d'Exact Match n'a été calculé.

## 14. Limites du système

Le projet est un POC présentant plusieurs limites :

- La recherche ne garantit pas qu'un événement propose une séance à une date précise.
- Les événements complets ne sont pas systématiquement exclus.
- Le système dépend d'une connexion Internet pour Mistral API.
- L'évaluation manuelle reste limitée.
- L'actualisation des données n'est pas automatisée.
- Il n'existe pas de mémoire de conversation entre les questions.

## 15. Perspectives d'amélioration

Les évolutions envisagées comprennent :

- Une gestion plus précise des dates et des horaires.
- La détection des événements complets ou annulés.
- Une mise à jour automatique des événements OpenAgenda.
- Une évaluation de la pertinence sur davantage de questions.
- L'amélioration du classement des événements.
- Le suivi des temps de réponse et des performances.

## Conclusion

Ce projet démontre la construction et la mise à disposition d'un chatbot RAG appliqué à la recommandation d'événements culturels.

Le système combine **Hugging Face, FAISS, LangChain et Mistral API**, puis expose ses fonctionnalités avec **FastAPI dans Docker**.

L'interface Streamlit permet de réaliser une démonstration interactive du système, de la question de l'utilisateur à la réponse générée et aux sources associées.