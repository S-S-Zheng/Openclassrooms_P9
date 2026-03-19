"""
tests/test_build_index.py

On vérifie que le lancement de la CLI d'indexation appelle bien la pipeline pour indexer:
``OpenAgendaFetcher`` -> ``EventDocumentProcessor`` -> ``EventRAGPipeline``.
"""

# imports
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from CLI.build_index import indexation

# ========================================


# ------------ Vérifie que la pipeline d'indexation est bien respectée
@pytest.mark.integration
@pytest.mark.asyncio
async def test_indexation_flow_success(fake_settings, raw_events, sample_documents):
    """Vérifie que la pipeline complète s'exécute dans l'ordre."""

    # On patch les CLASSES pour intercepter leur instanciation
    with (
        patch("CLI.build_index.get_settings", return_value=fake_settings),
        patch("CLI.build_index.OpenAgendaFetcher") as MockFetcherClass,
        patch("CLI.build_index.EventDocumentProcessor") as MockProcessorClass,
        patch("CLI.build_index.EventRAGPipeline") as MockRAGClass,
    ):
        # Config du Fetcher -> raw_events
        mock_fetcher_inst = MockFetcherClass.return_value
        mock_fetcher_inst.fetch_events = AsyncMock(return_value=raw_events)
        # Config du Processor -> sample_documents
        mock_processor_inst = MockProcessorClass.return_value
        mock_processor_inst.process = MagicMock(return_value=sample_documents)
        # Config du RAG -> on simule juste le succès des méthodes async concernant l'indexation
        mock_rag_inst = MockRAGClass.return_value
        mock_rag_inst.build_index = AsyncMock()
        mock_rag_inst.save_index = AsyncMock()
        mock_rag_inst.document_count = MagicMock(return_value=len(sample_documents))

        await indexation()

        # Assertions
        # Le fetcher a été appelé
        mock_fetcher_inst.fetch_events.assert_called_once()
        # Le processor a reçu EXACTEMENT les raw_events venant du fetcher
        mock_processor_inst.process.assert_called_once_with(raw_events)
        # Le RAG a reçu EXACTEMENT les documents sortant du processor
        mock_rag_inst.build_index.assert_called_once_with(sample_documents)
        # La sauvegarde a été déclenchée
        mock_rag_inst.save_index.assert_called_once()


# -------------- Vérifie que le programme s'arrête si pas d'event
@pytest.mark.integration
@pytest.mark.asyncio
async def test_indexation_exits_if_no_events(fake_settings):
    """Vérifie que le programme s'arrête proprement si aucun événement n'est trouvé."""
    with (
        patch("CLI.build_index.get_settings", return_value=fake_settings),
        patch("CLI.build_index.OpenAgendaFetcher") as MockFetcherClass,
        patch("CLI.build_index.EventDocumentProcessor") as MockProcessorClass,
        patch("CLI.build_index.EventRAGPipeline") as MockRAGClass,
        patch("CLI.build_index.sys.exit") as mock_exit,
    ):
        MockFetcherClass.return_value.fetch_events = AsyncMock(return_value=[])
        mock_exit.side_effect = SystemExit(1)

        with pytest.raises(SystemExit):
            await indexation()

        # Assertions
        # On vérifie que sys.exit(1) a été appelé
        mock_exit.assert_called_once_with(1)
        # Vérifier que le RAG et processor JAMAIS été appelé
        MockProcessorClass.return_value.process.assert_not_called()
        MockRAGClass.return_value.build_index.assert_not_called()
