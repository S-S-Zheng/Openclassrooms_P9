"""
app/api/schemas.py

Module de définition des schémas Pydantic pour l'API.\n
Ce module définit les contrats d'interface (Data Transfer Objects) pour les requêtes
et les réponses de l'API FastAPI.\n
Il inclut une couche de validation métier robuste pour garantir que les données envoyées
au service respectent les plages de valeurs attendues.

Schémas principaux:
    - SourceDocument : Schéma d'un document source extrait de la base de donnée vectorielle FAISS.
    - AskRequest : Question utilisateur avec nombre de document a récupérer par FAISS.
    - AskResponse : Rappel de la question, Réponse augmentée, sources et temps de latence.
    - RebuildResponse : Confirme la reconstruction de l'index FAISS.
"""

# Imports
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ========================= CHAT / INFERENCE (POST /ask) ===========================


class SourceDocument(BaseModel):
    """Représente un document source extrait de la base vectorielle FAISS."""

    content: str = Field(..., description="Extrait du contenu du document source.")
    event_title: str = Field(..., description="Titre de l'événement d'origine.")
    event_uid: str = Field(..., description="ID de l'événement.")
    score: Optional[float] = Field(
        default=0.0, description="Score de similarité cosinus (0–1) retourné par FAISS."
    )


class AskRequest(BaseModel):
    """Données d'entrée pour poser une question au RAG."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question en langage naturel sur les événements (3-1000 caractères)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Nombre de documents sources à récupérer dans FAISS (1-10)",
    )

    @field_validator("question")
    @classmethod
    def clean_question(cls, val: str) -> str:
        """Nettoie les espaces inutiles autour de la question."""
        return val.strip()

    model_config = ConfigDict(
        json_schema_extra={"example": {"question": "Un truc fun à faire?", "top_k": 1}}
    )


class AskResponse(BaseModel):
    """Schéma de réponse complet renvoyé après interrogation du RAG."""

    question: str = Field(..., description="La question telle qu'elle a été reçue.")
    answer: str = Field(..., description="Réponse par le LLM (Mistral) augmentée par le contexte.")
    sources: List[SourceDocument] = Field(
        ...,
        description="Liste ordonnée des sources utilisées pour générer la réponse.",
    )
    processing_time_ms: float = Field(
        ..., description="Temps total de traitement de la requête en millisecondes."
    )


# ========================= MAINTENANCE (POST /rebuild) ===========================


class RebuildResponse(BaseModel):
    """Schéma de réponse confirmant la reconstruction de l'index FAISS."""

    status: str = Field(..., description="Résultat de l'opération (ex: 'success').")
    documents_indexed: int = Field(
        ..., description="Nombre de documents ingérés et vectorisés dans FAISS."
    )
    duration_seconds: float = Field(..., description="Temps total de reconstruction en secondes.")
    rebuilt_at: datetime = Field(
        # La manière moderne et recommandée (Python 3.12+)
        # alternative: datetime.datetime.now(datetime.timezone.utc)
        default_factory=lambda: datetime.now(timezone.utc),
        description="Horodatage UTC du moment de la reconstruction.",
    )


# ========================= ERREURS & MÉTADONNÉES ===========================


class ErrorResponse(BaseModel):
    """Schéma standardisé pour les messages d'erreur de l'API (4xx, 5xx)."""

    error: str = Field(..., description="Type ou code de l'erreur.")
    detail: str = Field(..., description="Explication détaillée de la cause de l'erreur.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# class ModelInfoOutput(BaseModel):
#     """Métadonnées sur les modèles d'IA actuellement utilisés par le service."""

#     source: str = Field(..., description="Source de la documentation.")
#     llm_model: str = Field(..., description="Nom du modèle de génération utilisé.")
#     embed_model: str = Field(..., description="Nom du modèle d'embedding utilisé.")
#     vector_store: str = Field("FAISS", description="Type de base de données vectorielle.")
#     index_is_ready: bool = Field(..., description="Indique si l'index est chargé en mémoire.")
