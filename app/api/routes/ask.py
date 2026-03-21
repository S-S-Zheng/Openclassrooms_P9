"""
app/api/routes/ask.py

``POST /ask``
-------
Soumettre une question et recevoir une réponse générée par RAG.

La route est asynchrone: le travail lourd de LangChain / FAISS est déchargé dans un pool de threads
à l'intérieur de ``EventRAGPipeline.query()``pour ne pas bloquer la boucle d'événements.
"""

# Imports
import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_rag
from app.api.schemas import AskRequest, AskResponse, SourceDocument
from app.rag.base_rag import BaseRAG

# Nécéssaire qu'au point d'entrée donc main.py != avec CLI qui sont leur propre point d'entrée
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)-8s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
logger = logging.getLogger(__name__)
router = APIRouter(tags=["Ask"])


# =========================================================================


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    rag: Annotated[BaseRAG, Depends(get_rag)],
) -> AskResponse:
    """
    Poser une question et recevoir une réponse générée augmentée.

    Parameters
    ----------
    payload:
        Corps de la requête validé contenant la question et un ``top_k``.
    rag:
        Instance partagée de ``BaseRAG`` injectée par FastAPI.

    Returns
    -------
    AskResponse
        Réponse générée, documents sources et temps de traitement.
    """
    start_ms = time.monotonic()

    try:
        result = await rag.query(question=payload.question, top_k=payload.top_k)
    except RuntimeError as exc:
        # Index non prêt — ne devrait pas arriver si la dépendance get_rag est correcte,
        # mais nous protégeons de manière défensive.
        logger.error(f"La requête RAG a échoué : {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(f"Erreur inattendue pendant la requête: {payload.question}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Une erreur inattendue est survenue lors de la génération de la réponse.",
        ) from exc

    # Construction de la liste typée des documents sources pour la réponse
    sources: list[SourceDocument] = []
    # On itère sur les documents et leurs scores de similarité
    for doc, score in zip(result["source_documents"], result.get("scores", []), strict=False):
        title = doc.metadata.get("title")
        uid = doc.metadata.get("uid")

        # On n'ajoute au résultat que si les champs obligatoires sont présents
        if title and uid:
            sources.append(
                SourceDocument(
                    content=doc.page_content[:500],  # Aperçu des 500 premiers caractères
                    event_title=title,
                    event_uid=uid,
                    score=round(score, 4) if score is not None else 0.0,
                )
            )

    elapsed_ms = int((time.monotonic() - start_ms) * 1_000)
    logger.info(
        f"Question répondue en {elapsed_ms} ms | top_k={payload.top_k} | sources={len(sources)}"
    )

    return AskResponse(
        question=payload.question,
        answer=result["answer"],
        sources=sources,
        processing_time_ms=elapsed_ms,
    )
