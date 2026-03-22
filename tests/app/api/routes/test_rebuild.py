"""
tests/app/api/test_rebuild.py

Tests pour la route de reconstruction de l'index.
"""

# Imports
from unittest.mock import AsyncMock, patch

import pytest

# =======================================================


# --------------- Vérifie la pipeline de rebuild
@pytest.mark.unit
@patch("app.api.routes.rebuild.OpenAgendaFetcher")
@patch("app.api.routes.rebuild.EventDocumentProcessor")
async def test_rebuild_success(
    mock_processor_class,
    mock_fetcher_class,
    mock_client_ready,
    raw_events,
    sample_documents,
    fake_settings,
):
    """Vérifie ETL + Indexation."""
    # Configuration des Mocks
    # ******** Phase ETL
    # Le Fetcher renvoie nos raw_events
    mock_fetcher = mock_fetcher_class.return_value
    mock_fetcher.fetch_events = AsyncMock(
        return_value=raw_events
    )  # ``fetch_event()`` -> asynchrone
    # Le Processor renvoie nos sample_documents
    mock_processor = mock_processor_class.return_value
    mock_processor.process.return_value = sample_documents  # ``process()`` -> synchrone
    # ******** Phase index
    mock_rag = mock_client_ready.app.state.rag
    mock_rag.build_index = AsyncMock()
    mock_rag.save_index = AsyncMock()

    # Appel de la route
    mock_client_ready.app.state.settings = fake_settings
    headers = {"X-Rebuild-Key": fake_settings.rebuild_api_key}
    response = mock_client_ready.post("/rebuild", headers=headers)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["documents_indexed"] == len(sample_documents)
    # Vérification de l'orchestration
    mock_fetcher.fetch_events.assert_called_once()
    mock_processor.process.assert_called_once_with(raw_events)
    mock_rag.build_index.assert_called_once_with(sample_documents)
    mock_rag.save_index.assert_called_once()
    # Vérification du basculement d'état
    assert mock_client_ready.app.state.index_ready is True


# --------------- Vérifie que le rebuild ne se déclenche pas sans la clef
@pytest.mark.unit
def test_rebuild_forbidden_without_key(mock_client_ready):
    """Vérifie que la route est protégée si la clé API est absente ou incorrecte."""
    response = mock_client_ready.post("/rebuild")  # Pas de header
    assert response.status_code == 403
    assert "Clé API /rebuild invalide ou manquante." in response.json()["detail"]


# --------------- Vérifie le code 500 dans le cas où pas d'event récup
@pytest.mark.unit
@patch("app.api.routes.rebuild.OpenAgendaFetcher")
async def test_rebuild_error_no_events(mock_fetcher_class, mock_client_ready, fake_settings):
    """Vérifie l'erreur 500 si OpenAgenda ne renvoie rien."""
    mock_fetcher = mock_fetcher_class.return_value
    mock_fetcher.fetch_events = AsyncMock(return_value=[])  # Liste vide

    mock_client_ready.app.state.settings = fake_settings
    headers = {"X-Rebuild-Key": fake_settings.rebuild_api_key}
    response = mock_client_ready.post("/rebuild", headers=headers)

    # Assertions
    assert response.status_code == 500
    assert "Aucun événement récupéré" in response.json()["detail"]
