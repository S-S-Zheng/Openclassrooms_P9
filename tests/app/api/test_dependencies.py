"""
tests/app/api/test_dependencies.py

Tests unitaires pour la dépendance get_rag (le garde-fou).
"""

# imports
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request

from app.api.dependencies import get_rag

# ========================================================================


# ------------ Vérifie qu'on récupère une instance RAG opérationnelle
@pytest.mark.unit
def test_get_rag_success():
    """Vérifie que get_rag retourne l'instance si tout est prêt."""
    # Mock rag is_ready
    mock_rag = MagicMock()
    mock_rag.is_ready.return_value = True
    # Mock Request de FastAPI
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.rag = mock_rag

    # Appel de la dépendance
    result = get_rag(mock_request)

    # Assertions
    assert result == mock_rag
    mock_rag.is_ready.assert_called_once()


# ------------ Vérifie le code 503 (Pas de RAG)
@pytest.mark.unit
def test_get_rag_raises_503_if_none():
    """Vérifie l'erreur 503 si l'attribut rag n'existe pas dans l'état."""
    # Mock rag is_ready
    mock_rag = MagicMock()
    mock_rag.is_ready.return_value = True
    # Mock Request de FastAPI
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.rag = None  # pas de RAG

    with pytest.raises(HTTPException) as exc_info:
        get_rag(mock_request)

    # Assertions
    assert exc_info.value.status_code == 503
    assert "La pipeline RAG n'est pas initialisée." in exc_info.value.detail


# ------------ Vérifie le code 503 (Pas d'index)
@pytest.mark.unit
def test_get_rag_raises_503_if_not_ready():
    """Vérifie l'erreur 503 si le RAG existe mais que l'index n'est pas chargé."""
    # Mock rag is_ready
    mock_rag = MagicMock()
    mock_rag.is_ready.return_value = False  # Pas d'index
    # Mock Request de FastAPI
    mock_request = MagicMock(spec=Request)
    mock_request.app.state.rag = mock_rag

    with pytest.raises(HTTPException) as exc_info:
        get_rag(mock_request)

    # Assertions
    assert exc_info.value.status_code == 503
    assert "La pipeline RAG n'est pas prête. " in exc_info.value.detail
    assert "Aucun index FAISS trouvé, en attente d'un POST /rebuild." in exc_info.value.detail
