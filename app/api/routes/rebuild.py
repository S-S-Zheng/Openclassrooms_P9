"""
app/api/routes/rebuild.py

``POST /rebuild``
----------
récupère les nouveaux événements d'OpenAgenda, les vectorise (re-embed),
et remplace l'index FAISS actuel.

Sécurité
-------
La route est protégée par la dépendance ``require_rebuild_key``
(en-tête ``X-Rebuild-Key``). Une clé incorrecte ou manquante renvoie un HTTP 403.
"""

# imports
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.schemas import RebuildResponse
from app.core.security import require_rebuild_key
from app.etl.fetcher import OpenAgendaFetcher
from app.etl.processor import EventDocumentProcessor
from app.rag.base_rag import BaseRAG

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Rebuild"])


# ===================================================================


@router.post(
    "/rebuild",
    # Modèle de réponse pour la documentation automatique
    # Barrière de sécurité (API Key)
    response_model=RebuildResponse,
    dependencies=[Depends(require_rebuild_key)],
)
async def rebuild(request: Request) -> RebuildResponse:
    """
    Orchestrer une reconstruction complète de l'index.

    Parameters
    ----------
    request:
        L'objet ``Request`` actuel, utilisé pour accéder à ``app.state`` (RAG + paramètres).

    Returns
    -------
    RebuildResponse
        Nombre de documents indexés et durée totale.
    """
    settings = request.app.state.settings
    rag: BaseRAG = request.app.state.rag

    start_ms = time.monotonic()
    logger.info("Début de la reconstruction de l'index...")

    try:
        # Récupération des événements bruts
        fetcher = OpenAgendaFetcher(settings)
        raw_events = await fetcher.fetch_events()
        logger.info(f"{len(raw_events)} événements bruts récupérés.")

        if not raw_events:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Aucun événement récupéré.",
            )

        # Transformation en Documents LangChain
        processor = EventDocumentProcessor(settings)
        documents = processor.process(raw_events)
        logger.info(f"{len(documents)} documents traités.")

        # Build et Save
        await rag.build_index(documents)
        await rag.save_index()

    except HTTPException:
        raise  # On renvoie tel quel l'erreur 500 levée ci-dessus
    except Exception as exc:
        logger.exception("Échec du rebuild.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur Rebuild: {exc}",
        ) from exc

    # On marque l'index comme prêt pour les requêtes /ask
    request.app.state.index_ready = True
    duration = round(time.monotonic() - start_ms, 2)
    logger.info(
        "Service basculé en mode opérationnel."
        f"Temps d'exec: {duration} | docs indexés: {len(documents)}"
    )

    return RebuildResponse(
        status="success",
        documents_indexed=len(documents),
        duration_seconds=duration,
        rebuilt_at=datetime.now(tz=timezone.utc),
    )
