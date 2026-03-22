"""
tests/test_qa_helper.py

On vérifie que qa_helper parvient à extraire et afficher les documents de rag._vecstore
de façon random (deterministe)
"""

# Imports
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from CLI.qa_helper import inspect_index

# =====================================================================


# ----------- Vérifie l'attribut interne ``_dict()`` du docstore si possible
@pytest.mark.unit
@pytest.mark.asyncio
async def test_inspect_index_with_docstore_dict(capsys, fake_settings):
    """
    Vérifie l'extraction quand le docstore possède un attribut _dict
    (cas classique InMemoryDocstore).
    """
    # Préparation des faux documents
    mock_doc = MagicMock()
    mock_doc.metadata = {"title": "Festival Jazz", "uid": "123"}
    mock_doc.page_content = "Un super festival de musique à Paris."

    # Mock de la pipeline RAG et de sa structure interne
    with patch("CLI.qa_helper.EventRAGPipeline") as MockRAG:
        instance = MockRAG.return_value
        instance.load_index = AsyncMock(return_value=True)
        # Simulation de rag._vectorstore.docstore._dict
        mock_vectorstore = MagicMock()
        mock_docstore = MagicMock()
        # On simule 5 documents pour que random.sample(..., 5) ne crash pas
        mock_docstore._dict = {f"id_{i}": mock_doc for i in range(5)}

        mock_vectorstore.docstore = mock_docstore
        instance._vectorstore = mock_vectorstore

        # Exécution
        await inspect_index()

        # Assertions
        captured = capsys.readouterr()
        assert "Festival Jazz" in captured.out
        assert "ID: 123" in captured.out
        assert "----- Extrait -----:\nUn super festival" in captured.out
        assert "-" * 50 in captured.out


# ----------- Vérifie le fallback sur les ID si ``_dict()`` n'existe pas
@pytest.mark.unit
@pytest.mark.asyncio
async def test_inspect_index_fallback_logic(capsys):
    """
    Vérifie la logique de repli (fallback) si _dict n'existe pas
    en utilisant search() et index_to_docstore_id.
    """
    mock_doc = MagicMock()
    mock_doc.metadata = {"title": "Festival Jazz", "uid": "123"}
    mock_doc.page_content = "Un super festival de musique à Paris."

    with patch("CLI.qa_helper.EventRAGPipeline") as MockRAG:
        instance = MockRAG.return_value
        instance.load_index = AsyncMock(return_value=True)
        mock_vectorstore = MagicMock()
        mock_docstore = MagicMock()

        # On supprime l'attribut _dict pour forcer le fallback
        if hasattr(mock_docstore, "_dict"):
            del mock_docstore._dict
        # Simulation du mapping d'index FAISS vers IDs docstore
        mock_vectorstore.index_to_docstore_id = {i: f"id_{i}" for i in range(5)}
        # Simulation de la méthode search du docstore
        mock_docstore.search.return_value = mock_doc

        mock_vectorstore.docstore = mock_docstore
        instance._vectorstore = mock_vectorstore

        await inspect_index()

        # Assertions
        captured = capsys.readouterr()
        assert "Festival Jazz" in captured.out
        assert mock_docstore.search.call_count == 5


# ----------- Vérifie qu'un index vide chargé renvoi le bon message et ne plante pas
@pytest.mark.unit
@pytest.mark.asyncio
async def test_inspect_index_empty(capsys):
    """Vérifie le message d'erreur si aucun document n'est trouvé."""
    with patch("CLI.qa_helper.EventRAGPipeline") as MockRAG:
        instance = MockRAG.return_value
        instance.load_index = AsyncMock(return_value=True)

        instance._vectorstore.docstore._dict = {}  # Index vide

        await inspect_index()

        # Assertions
        captured = capsys.readouterr()
        assert "L'index est chargé mais ne contient aucun document." in captured.out
