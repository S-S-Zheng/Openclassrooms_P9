"""
app/api/dependencies.py

Fournisseurs de dépendances pour FastAPI. Aura pour rôle de :
* Extraire des informations des requêtes.
* Valider l'état du système avant d'atteindre les routes.
* Gérer les erreurs HTTP (401, 403, 503).

Exemples de fonctions pouvant y être mis mettre:
* get_current_user (pour l'authentification).
* get_db (pour la session de base de données).
Ne se met pas dans utils/ mais plutôt ici car intrinsèquement lié a FastAPI
car il utilise des objets spécifiques comme Request, HTTPException


"""

# imports
from fastapi import HTTPException, Request, status

from app.rag.base_rag import BaseRAG

# ========================================================================


def get_rag(request: Request) -> BaseRAG:
    """
    Dépendance FastAPI fournissant l'instance partagée de ``BaseRAG``.\n
    L'instance est stockée dans ``app.state.rag`` par le gestionnaire de contexte
    ``lifespan`` dans ``app/main.py``. Va servir de bloqueur d'appel tant que la pipeline RAG
    n'est pas opérationnelle.

    Avantages :
    * Fail-Fast (Échec immédiat)
        Au lieu de laisser la requête entrer dans la logique complexe du RAG et de gaspiller
        des ressources (ou de générer des logs d'erreurs cryptiques), on bloque à l'entrée.
    * Séparation des responsabilités
        Les routes (comme /ask) n'ont pas besoin de vérifier si le RAG est prêt.
        Elles reçoivent un objet rag et elles savent qu'il est opérationnel.
        Cela rend le code des routes beaucoup plus propre.
    * Expérience Utilisateur (UX)
        Le message d'erreur est explicite. L'utilisateur (ou le développeur du front-end)
        sait exactement pourquoi ça ne marche pas : "Il manque l'index, faire un /rebuild".

    Parameters
    ----------
    request:
        Objet ``Request`` actuel de FastAPI (injecté automatiquement).

    Returns
    -------
    BaseRAG
        L'instance de la pipeline RAG à l'échelle de l'application.

    Raises
    ------
    HTTPException
        HTTP 503 si la pipeline RAG n'est pas encore prête (index non chargé).
    """
    # Récupération de l'instance RAG stockée dans l'état de l'application
    rag: BaseRAG | None = getattr(request.app.state, "rag", None)

    # Cas 1 : L'objet RAG n'existe même pas (erreur d'initialisation grave)
    if rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La pipeline RAG n'est pas initialisée.",
        )

    # Cas 2 : L'objet existe mais l'index FAISS n'est pas chargé (is_ready() == False)
    if not rag.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "La pipeline RAG n'est pas prête. "
                "Aucun index FAISS trouvé, en attente d'un POST /rebuild."
            ),
        )

    return rag
