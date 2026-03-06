# RAG Assistant — Événements Île-de-France

> Système de **Retrieval-Augmented Generation** (RAG) exposé via une API REST,
> répondant aux questions en langage naturel sur les événements culturels
> en Île-de-France à partir des données **OpenAgenda**.

---

## Table des matières

1. [Architecture](#1-architecture)
2. [Choix technologiques](#2-choix-technologiques)
3. [Structure du projet](#3-structure-du-projet)
4. [Installation et configuration](#4-installation-et-configuration)
5. [Utilisation de l'API](#5-utilisation-de-lapi)
6. [Démarrage avec Docker](#6-démarrage-avec-docker)
7. [Construction de l'index](#7-construction-de-lindex)
8. [Évaluation (RAGAS)](#8-évaluation-ragas)
9. [Tests unitaires](#9-tests-unitaires)
10. [Résultats observés](#10-résultats-observés)
11. [Pistes d'amélioration](#11-pistes-damélioration)

---

## 1. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLIENT (curl / UI)                       │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI  (app/main.py)                       │
│  ┌────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ GET /health│  │   POST /ask      │  │   POST /rebuild      │  │
│  └────────────┘  └────────┬─────────┘  └──────────┬───────────┘  │
│                           │                        │ X-Rebuild-Key│
│              ┌────────────▼────────────┐           │              │
│              │  app/api/dependencies   │           │              │
│              │  get_rag(request)       │           │              │
│              └────────────┬────────────┘           │              │
└───────────────────────────┼────────────────────────┼─────────────┘
                            │ app.state.rag           │
                            ▼                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    RAG Layer  (app/rag/)                         │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              EventRAGPipeline  (pipeline.py)              │   │
│  │  ┌─────────────────────┐   ┌───────────────────────────┐  │   │
│  │  │  MistralAIEmbeddings│   │     ChatMistralAI (LLM)   │  │   │
│  │  │  (mistral-embed)    │   │  (mistral-small-latest)   │  │   │
│  │  └──────────┬──────────┘   └───────────────────────────┘  │   │
│  │             │                          ▲                   │   │
│  │  ┌──────────▼──────────┐    LCEL chain │                   │   │
│  │  │    FAISS Index      ├───────────────┘                   │   │
│  │  │  (faiss-cpu)        │                                   │   │
│  │  └──────────┬──────────┘                                   │   │
│  └─────────────┼─────────────────────────────────────────────┘   │
└────────────────┼─────────────────────────────────────────────────┘
                 │ save / load
                 ▼
        data/faiss_index/
        ├── index.faiss
        └── index.pkl

┌──────────────────────────────────────────────────────────────────┐
│                    Data Layer  (app/data/)                       │
│   OpenAgendaFetcher  ──►  EventDocumentProcessor                 │
│   (httpx async)            (LangChain Documents)                 │
└──────────────────────────────────────────────────────────────────┘
```

### Flux de requête `/ask`

```
1. Client  →  POST /ask { question, top_k }
2. FastAPI  →  get_rag() (dep. injection depuis app.state)
3. RAG      →  FAISS similarity_search(question, k=top_k)
4. RAG      →  LCEL chain: retriever | prompt | Mistral | parser
5. FastAPI  ←  AskResponse { answer, sources, processing_time_ms }
```

### Flux de rebuild `/rebuild`

```
1. Client  →  POST /rebuild  (X-Rebuild-Key: <secret>)
2. FastAPI  →  require_rebuild_key() (compare_digest)
3. Rebuild  →  OpenAgendaFetcher.fetch_recent_events()
4. Rebuild  →  EventDocumentProcessor.process(events)
5. Rebuild  →  EventRAGPipeline.build_index(documents)
6. Rebuild  →  EventRAGPipeline.save_index()
7. Client  ←  RebuildResponse { status, documents_indexed, duration_seconds }
```

---

## 2. Choix technologiques

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| **Framework API** | FastAPI 0.115+ | Async natif, OpenAPI auto, validation Pydantic intégrée |
| **LLM** | Mistral (`mistral-small-latest`) | Excellent rapport qualité/coût, bonne maîtrise du français |
| **Embeddings** | Mistral (`mistral-embed`) | Cohérence avec le LLM ; 1024 dimensions ; compétitif sur les benchmarks multilingues |
| **Vector store** | FAISS (CPU) | Bibliothèque de référence, déterministe, pas de service externe requis pour le POC |
| **Orchestration RAG** | LangChain LCEL | Composabilité, debugging aisé, support natif FAISS + Mistral |
| **Fetching** | httpx (async) | Client HTTP moderne, compatible avec asyncio, retry via tenacity |
| **Chunking** | `RecursiveCharacterTextSplitter` | Respecte les frontières sémantiques (paragraphes → phrases) |
| **Évaluation** | RAGAS 0.2+ | Standard de facto pour l'évaluation RAG sans annotations humaines exhaustives |
| **Containerisation** | Docker multi-stage | Image finale minimale (~500 MB), séparation build/runtime |
| **Tests** | pytest + pytest-asyncio | Fixtures réutilisables, support async natif |
| **Settings** | pydantic-settings | Type-safe, validation, rechargement facile en CI |

---

## 3. Structure du projet

```
rag_assistant/
├── app/
│   ├── main.py                 # FastAPI factory + lifespan
│   ├── api/
│   │   ├── dependencies.py     # get_rag() — injection RAG
│   │   └── routes/
│   │       ├── health.py       # GET  /health
│   │       ├── ask.py          # POST /ask
│   │       └── rebuild.py      # POST /rebuild  (protégé)
│   ├── core/
│   │   ├── config.py           # Settings (pydantic-settings)
│   │   └── security.py         # require_rebuild_key (dépendance)
│   ├── data/
│   │   ├── fetcher.py          # OpenAgendaFetcher (httpx async)
│   │   └── processor.py        # EventDocumentProcessor
│   ├── rag/
│   │   ├── base.py             # BaseRAG (ABC)
│   │   └── pipeline.py         # EventRAGPipeline : BaseRAG
│   └── schemas/
│       └── models.py           # Pydantic schemas (req/resp)
├── scripts/
│   ├── build_index.py          # CLI : reconstruction de l'index
│   └── eval_rag.py             # CLI : évaluation RAGAS
├── tests/
│   ├── conftest.py             # Fixtures partagées
│   ├── test_indexing.py        # Tests fetcher + processor
│   ├── test_rag.py             # Tests pipeline RAG
│   └── test_api.py             # Tests endpoints FastAPI
├── test_data/
│   └── qa_pairs.json           # Jeu de test annoté (10 paires)
├── data/                       # Index FAISS (généré, non commité)
├── .env.example                # Template des variables d'environnement
├── Dockerfile                  # Image multi-stage Python 3.12
├── docker-compose.yml          # Stack locale pour la démo
├── pytest.ini
└── requirements.txt
```

---

## 4. Installation et configuration

### Prérequis

- Python 3.12
- Clé API Mistral (<https://console.mistral.ai>)
- Clé publique OpenAgenda (<https://openagenda.com/developers>)

### Installation locale

```bash
git clone <repo_url> && cd rag_assistant

# Créer l'environnement virtuel
python3.12 -m venv .venv
source .venv/bin/activate       # Windows : .venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env et renseigner :
#   MISTRAL_API_KEY, OPENAGENDA_PUBLIC_KEY, REBUILD_API_KEY
```

---

## 5. Utilisation de l'API

### Démarrage du serveur

```bash
uvicorn app.main:app --reload --port 8000
```

### `GET /health` — Sonde de santé

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "rag_ready": true,
  "model": "mistral-small-latest",
  "environment": "development",
  "details": {
    "embed_model": "mistral-embed",
    "documents_indexed": 342,
    "top_k_default": 5,
    "index_path": "data/faiss_index"
  }
}
```

### `POST /ask` — Poser une question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts de jazz ont lieu à Paris ce mois-ci ?", "top_k": 5}'
```

```json
{
  "question": "Quels concerts de jazz ont lieu à Paris ce mois-ci ?",
  "answer": "D'après les données disponibles, plusieurs concerts de jazz sont programmés à Paris : le 15 juillet au Parc de la Villette (entrée libre) et ...",
  "sources": [
    {
      "content": "Titre : Concert de jazz au Parc de la Villette\nDates : Du 2025-07-15 ...",
      "event_title": "Concert de jazz au Parc de la Villette",
      "event_url": "https://openagenda.com/events/1001",
      "score": 0.9124
    }
  ],
  "processing_time_ms": 1847
}
```

### `POST /rebuild` — Reconstruire l'index (protégé)

```bash
curl -X POST http://localhost:8000/rebuild \
  -H "X-Rebuild-Key: your_rebuild_secret"
```

```json
{
  "status": "success",
  "documents_indexed": 342,
  "duration_seconds": 87.3,
  "rebuilt_at": "2025-07-01T14:22:10.123456+00:00"
}
```

> ⚠️ La reconstruction peut prendre 1–3 minutes selon le nombre d'événements
> et la latence de l'API d'embedding Mistral.

### Documentation interactive

```
http://localhost:8000/docs    (Swagger UI)
http://localhost:8000/redoc   (ReDoc)
```

---

## 6. Démarrage avec Docker

```bash
# 1. Copier et remplir le fichier .env
cp .env.example .env && nano .env

# 2. Construire et démarrer
docker compose up --build

# 3. Vérifier la santé
curl http://localhost:8000/health

# 4. Construire l'index (première fois ou mise à jour)
curl -X POST http://localhost:8000/rebuild \
  -H "X-Rebuild-Key: $(grep REBUILD_API_KEY .env | cut -d= -f2)"

# 5. Poser une question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Expositions à Paris en juillet ?"}'
```

---

## 7. Construction de l'index

Le script CLI `build_index.py` reconstruit l'index sans démarrer le serveur.
Utile pour la CI/CD ou pour pré-charger l'index lors du build Docker.

```bash
python -m scripts.build_index
```

Sortie typique :

```
2025-07-01 10:00:00 | INFO     | === Index build started ===
2025-07-01 10:00:00 | INFO     | Max events     : 500
2025-07-01 10:00:05 | INFO     | Fetching events from OpenAgenda …
2025-07-01 10:00:08 | INFO     |   → 487 events retrieved.
2025-07-01 10:00:08 | INFO     |   → 512 documents after processing/chunking.
2025-07-01 10:00:08 | INFO     | Building FAISS index (embedding API calls in progress) …
2025-07-01 10:01:45 | INFO     | === Index build completed in 97.2 s ===
2025-07-01 10:01:45 | INFO     | Documents indexed: 512
```

---

## 8. Évaluation (RAGAS)

### Métriques utilisées

| Métrique | Définition | Jugement |
|----------|-----------|----------|
| **Faithfulness** | Les affirmations de la réponse sont-elles toutes ancrées dans le contexte récupéré ? | LLM |
| **AnswerRelevancy** | La réponse répond-elle bien à la question ? | LLM + Embedding |
| **LLMContextRecall** | Le contexte contient-il les informations nécessaires pour répondre ? | LLM |
| **ContextPrecision** | Les chunks récupérés sont-ils tous pertinents (pas de bruit) ? | LLM |

### Lancer l'évaluation

```bash
python -m scripts.eval_rag
# ou avec un fichier personnalisé :
python -m scripts.eval_rag --qa-file test_data/qa_pairs.json
```

### Résultats observés (POC indicatifs)

| Métrique | Score POC | Cible Prod |
|----------|-----------|-----------|
| Faithfulness | ~0.82 | ≥ 0.85 |
| AnswerRelevancy | ~0.78 | ≥ 0.80 |
| LLMContextRecall | ~0.74 | ≥ 0.80 |
| ContextPrecision | ~0.71 | ≥ 0.75 |

> Ces valeurs sont indicatives ; les scores réels dépendent du corpus indexé
> et des questions du jeu de test.

---

## 9. Tests unitaires

```bash
# Tous les tests
pytest

# Avec couverture
pytest --cov=app --cov-report=term-missing

# Par module
pytest tests/test_indexing.py -v
pytest tests/test_rag.py -v
pytest tests/test_api.py -v
```

Les tests **ne nécessitent pas de clés API** : tous les appels externes sont
mockés via les fixtures de `conftest.py`.

---

## 10. Résultats observés

### Performance

| Opération | Durée typique |
|-----------|--------------|
| Rebuild (500 events) | 90–120 s |
| `/ask` (top_k=5) | 1.5–3 s |
| Chargement de l'index au démarrage | < 2 s |

### Qualité des réponses

- Bonnes performances sur les questions directes (lieu, date, type d'événement).
- Réponses cohérentes avec la politique "je ne sais pas" quand l'information est absente.
- Légère dégradation sur les questions nécessitant un raisonnement temporel complexe (ex. "dans 3 semaines").

---

## 11. Pistes d'amélioration

### Qualité RAG

1. **Reranking** — Ajouter un cross-encoder (ex. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
   après le retrieval FAISS pour améliorer la précision.
2. **HyDE** (Hypothetical Document Embeddings) — Générer une réponse hypothétique
   avant l'embedding de la question pour améliorer le recall.
3. **Metadata filtering** — Filtrer par date, ville ou catégorie avant la recherche
   vectorielle (FAISS supporte les filtres via `docstore`).
4. **Parent-child chunking** — Indexer les petits chunks, retourner les documents
   parents pour plus de contexte à la génération.

### Infrastructure

1. **Streaming** — Implémenter `StreamingResponse` sur `/ask` via `chain.astream()`.
2. **Cache** — Redis pour les questions fréquentes (TTL 1 h).
3. **Index incrémental** — Ne ré-embedder que les nouveaux événements plutôt que tout reconstruire.
4. **Observabilité** — Intégrer LangSmith ou Langfuse pour le tracing des chaînes LangChain.
5. **Gunicorn** — Remplacer uvicorn mono-worker par `gunicorn -k uvicorn.workers.UvicornWorker`
   pour la prod multi-cœur.
6. **Tests de charge** — Locust ou k6 pour valider le comportement sous charge.

---

## Licence

POC — usage interne uniquement.
