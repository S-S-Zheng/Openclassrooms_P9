"""
app/main.py

Point d'entrée principal de l'application FastAPI.

Ce module assemble les différents composants de l'architecture :
1. Orchestre le cycle de vie de l'application (Lifespan) pour le chargement du modèle.
2. Centralise l'inclusion des routeurs.
3. Définit les endpoints de base comme la vérification de l'état (Healthcheck).

Notes:
-------
L'utilisation de ``@asynccontextmanager`` + ``lifespan=`` (FastAPI ≥ 0.93) permettent:
* De charger entièrement le RAG AVANT que le serveur ne se mettent a accepter des requetes.
* On protège le service en démarrant en mode dégradé si l'index n'est pas encore initialisé et
    ``get_rag()`` renverra un code statut 503 jusqu'au lancement de ``/rebuild``
* Les ressources occupées par le service sont correctement déchargées à l'extinction du serveur.

Fonctionnement de la boucle utilisateur-RAG (question NLP -> Réponse NLP):
* L'utilisateur envoi une requête /POST ``/ask``.
* FastAPI injecte la dépendance rag ``EventRAGPipeline``(stockée dans app.state).
* La pipeline utilise LangChain pour transformer la question en vecteur et FAISS cherche les
    événements les plus proches par recherche de similarité avec scoring.
* Un dictionnaire est retourné et transmis au LLM (Mistral) qui renvoi la réponse finale NLP.
"""

# Imports
import logging

# import logging.config
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse

from app.core.config import get_settings
from app.rag.rag_pipeline import EventRAGPipeline

# from app.api.routes.ask import router as aks_router
# from app.api.routes.rebuild import router as rebuild_router

# ======================= Logging configuration =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================= Lifespan ===============================================
# assynccontextmanager est un décorateur qui permet de définir une fonction
# capable de gérer une phase avant de démarrage et une après d'arrêt.
# Ici, tout ce qui est écrit avant yield s'éxé UNE SEULE FOIS au lancement
# du serveur ce qui permet de maintenit l'état tant que le serveur est en ON
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Gère le cycle de vie de l'application (Démarrage et Arrêt).

    Tout le code situé avant l'instruction 'yield' est exécuté une seule fois
    lors du lancement du serveur. Cela permet d'initialiser les ressources lourdes
    et de les maintenir en mémoire RAM tout au long de la session.

    Actions au démarrage :
        - Cache les settings
        - Instancie le rag (``EventRAGPipeline``)
        - Tente de charger un index FAISS depuis le disque si possible
            sinon démarre en dégradé (code HTTP 503)
        - Injection de ``settings`` et ``rag`` dans ``app.state`` pour un accès global
            via les requêtes.

    Args:
        app (FastAPI): L'instance de l'application.
    """

    # ============== Phase de démarrage ================
    # Cache singleton
    settings = get_settings()

    # Rappel log
    logger.info(
        f"Démarrage du système RAG: env = {settings.app_env}, LLM = {settings.llm_model}"
        f"Embedding = {settings.embed_model}..."
    )

    # Instancie la pipeline RAG (pas encore d'appel API)
    rag = EventRAGPipeline(settings)

    # Tentative de chargement de l'index FAISS depuis le disque
    loaded = await rag.load_index()
    if loaded:
        logger.info("Index FAISS chargé, service opérationnel")
    else:
        # Au lieu de faire planter le serveur si l'index FAISS est absent,
        # log un warning et laisses le serveur démarrer.
        # C'est crucial pour que l'endpoint /rebuild puisse justement être appelé pour
        # corriger le problème.
        logger.warning("Pas d'index trouvé, démarrage en mode dégradé (HTTP 503 sur /ask)")

    # Stockage settings et RAG pour accessible partout
    app.state.settings = settings
    app.state.rag = rag

    yield  # le serveur accepte les requêtes à partir d'ici

    # ================== Phase d'arrêt =================
    logger.info("Arrêt du serveur, nettoyage...")


# ================= Montage des Routers ==================================
# Instancie un router spécific pour les routes par défaut
generic_router = APIRouter()


# /health
# Test auto CI/CD, debug rapide
# FONDAMENTAL + NE DOIT JAMAIS DEPENDRE DE QUOIQUE CE SOIT
@generic_router.get("/health", tags=["Health"])
async def healthcheck():
    """
    Vérifie la disponibilité opérationnelle du service.\n
    Ce endpoint est crucial pour les outils de monitoring.
    Il doit rester indépendant des ressources externes pour
    isoler les pannes réseau/modèle de la panne serveur.

    Returns:
        dict: Un dictionnaire indiquant le statut opérationnel.
    """
    return {"status": "ok"}


# / (root)
# Feedback immédiat, debug, UX minimale
@generic_router.get("/", tags=["Root"], include_in_schema=False)
async def root():
    """
    Point d'entrée racine.\n
    Redirige automatiquement l'utilisateur vers la documentation Swagger
    interactive (/docs) pour faciliter l'exploration de l'API.

    Returns:
        RedirectResponse: Redirection vers l'interface utilisateur Swagger.
    """
    return RedirectResponse(url="/docs")


# ======================= API factory ==============================================
# factory `create_app()` plutôt que `app = FastAPI()` directement car les tests appellent
# `create_app()` pour créer une instance nouvelle de l'application pour chaque test.
# Sans factory, tous les tests partageraient la même instance.
def create_app() -> FastAPI:
    """
    Créée et configure l'app FastAPI.

    Returns
    -------
    FastAPI
    """
    settings = get_settings()

    app = FastAPI(
        title="Système RAG evènementiel orienté IDF",
        description=(
            "Chatbot intelligent s'appuyant sur un système RAG "
            "(LangChain + Mistral + FAISS) pour répondre aux questions "
            "sur les événements culturels en Île-de-France."
        ),
        version="1.0.0",
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.app_env != "production" else None,
        lifespan=lifespan,
    )

    # Applique la configuration des routes
    app.include_router(generic_router)
    # app.include_router(ask_router)
    # app.include_router(rebuild_router)

    return app


# ===========================================================================


# Module-level app instance used by uvicorn / gunicorn
app = create_app()
