"""
tests/app/api/test_ask.py

Tests sur la route ask.
"""

# Imports
from unittest.mock import AsyncMock, MagicMock

import pytest

# ===================================================================
# On nu'tilise que fake_llm_response et pas sample_document dans les fixtures car
# sample est lourd et doit être modifié dans les tests


# ------------- Happy path
@pytest.mark.unit
def test_ask_success(mock_client_ready, fake_llm_response):
    """Vérifie le bon déroulement d'une requête RAG réussie."""
    mock_rag = mock_client_ready.app.state.rag

    # On simule un document LangChain avec ses métadonnées
    mock_doc = MagicMock()
    mock_doc.page_content = "Blabla" * 10
    mock_doc.metadata = {"title": "test_ask_success", "uid": "777"}

    mock_rag.query = AsyncMock(
        return_value={
            "answer": fake_llm_response,
            "source_documents": [mock_doc],
            "scores": [0.98764321],
        }
    )

    # Exécution de la requête
    payload = {"question": "Quel est l'événement ?", "top_k": 1}
    response = mock_client_ready.post("/ask", json=payload)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == fake_llm_response
    assert len(data["sources"]) == 1
    assert data["sources"][0]["event_title"] == "test_ask_success"
    assert data["sources"][0]["score"] == 0.9876  # Vérifie l'arrondi (round 4)
    assert data["processing_time_ms"] >= 0


# ------------- Vérifie l'activité du code 500 (crash RAG)
@pytest.mark.unit
def test_ask_unexpected_error(mock_client_ready, fake_llm_response):
    """Vérifie que la route renvoie une 500 en cas de crash du moteur RAG."""
    mock_rag = mock_client_ready.app.state.rag

    # On simule un document LangChain avec ses métadonnées
    mock_doc = MagicMock()
    mock_doc.page_content = "Blabla" * 10
    mock_doc.metadata = {"title": "test_ask_success", "uid": "777"}

    mock_rag.query = AsyncMock(
        return_value={
            "answer": fake_llm_response,
            "source_documents": [mock_doc],
            "scores": [0.98764321],
        }
    )
    # On force une erreur catastrophique
    mock_rag.query.side_effect = Exception("Test crash RAG...")

    response = mock_client_ready.post("/ask", json={"question": "Quel est l'événement ?"})

    # Assertions
    assert response.status_code == 500
    assert "Une erreur inattendue est survenue lors de la génération" in response.json()["detail"]


# ------------ Vérifie le filtrage des documents difformes
@pytest.mark.unit
def test_ask_filters_malformed_sources(mock_client_ready, fake_llm_response):
    """Vérifie que les documents sans title ou uid sont ignorés."""
    mock_rag = mock_client_ready.app.state.rag

    bad_doc = MagicMock()
    bad_doc.metadata = {"title": None}  # Pas de titre

    mock_rag.query = AsyncMock(
        return_value={"answer": fake_llm_response, "source_documents": [bad_doc], "scores": [0.1]}
    )
    response = mock_client_ready.post("/ask", json={"question": "Quel est l'événement ?"})

    # Assertions
    # La réponse doit être 200, mais la liste des sources doit être vide
    assert response.status_code == 200
    assert len(response.json()["sources"]) == 0
