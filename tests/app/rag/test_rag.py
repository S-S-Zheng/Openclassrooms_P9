"""
tests/test_rag.py

Tous les appels API Mistral (embeddings + LLM) sont interceptés par la fixture
``rag_pipeline_with_index`` dans conftest.py, qui injecte un
véritable index FAISS construit à partir d'embeddings fictifs mais déterministes.
"""

# imports
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.rag.rag_pipeline import EventRAGPipeline

# ========================================================================


class TestEventRAGPipeline:
    # ======================== ``is_ready()`` =================================

    # ---------------- Vérifie que ``is_ready()`` ne l'est pas sans vérification d'index
    @pytest.mark.unit
    def test_not_ready_before_index(self, fake_settings):
        """Un pipeline fraîchement instancié ne doit pas être prêt."""
        with (
            patch("app.rag.rag_pipeline.MistralAIEmbeddings"),
            patch("app.rag.rag_pipeline.ChatMistralAI"),
        ):
            pipeline = EventRAGPipeline(fake_settings)

        assert pipeline.is_ready() is False
        assert pipeline.document_count() == 0

    # ---------------- Vérifie que ``is_ready()`` l'est après indexation
    @pytest.mark.unit
    def test_ready_after_index_injected(self, rag_pipeline_with_index):
        """Un pipeline avec un store FAISS injecté doit être déclaré prêt."""
        assert rag_pipeline_with_index.is_ready() is True

    # ---------------- Tests pour la gestion de l'état de ``document_count()``
    @pytest.mark.unit
    def test_document_count_matches_fixture(self, rag_pipeline_with_index, sample_documents):
        """``document_count()`` doit être égal au nombre de documents injectés."""
        assert rag_pipeline_with_index.document_count() == len(sample_documents)

    # ======================== ``_format_docs()`` ====================================

    # ---------------- Vérifie la sortie de ``_format_docs()``
    @pytest.mark.unit
    def test_format_docs_output(self, sample_documents):
        """Vérifie que le formatage des documents pour le prompt est correct."""
        from app.rag.rag_pipeline import EventRAGPipeline

        # On utilise les documents de ta fixture
        formatted_text = EventRAGPipeline._format_docs(sample_documents)

        # Vérifications
        assert "Résultat #1" in formatted_text
        assert "Résultat #2" in formatted_text
        # Vérifie que le contenu est présent
        assert sample_documents[0].page_content in formatted_text
        # Vérifie que l'URL (source) est bien extraite
        assert f"Source: {sample_documents[0].metadata['url']}" in formatted_text
        # Vérifie le double saut de ligne entre les sections
        assert "\n\n" in formatted_text

    # ======================= ``build_index()`` ===============================

    # ---------------- Vérifie que la pipeline est opérationnelle après ``build_index()``
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_build_index_sets_ready(self, fake_settings, sample_documents, fake_embeddings):
        """Après ``build_index()``, le pipeline doit être prêt."""
        with (
            patch("app.rag.rag_pipeline.MistralAIEmbeddings", return_value=fake_embeddings),
            patch("app.rag.rag_pipeline.ChatMistralAI"),
            # Patch FAISS.from_documents to avoid calling the real embedding API
            patch(
                "app.rag.rag_pipeline.FAISS.from_documents",
                return_value=MagicMock(
                    as_retriever=MagicMock(return_value=MagicMock()),
                    index=MagicMock(ntotal=len(sample_documents)),
                ),
            ),
        ):
            pipeline = EventRAGPipeline(fake_settings)
            await pipeline.build_index(sample_documents)

        assert pipeline.is_ready() is True
        assert pipeline.document_count() == len(sample_documents)

    # ---------------- Vérifie qu'une erreur est levée si vide
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_build_index_raises_on_empty(self, fake_settings):
        """``build_index([])`` doit lever une ``ValueError``."""
        with (
            patch("app.rag.rag_pipeline.MistralAIEmbeddings"),
            patch("app.rag.rag_pipeline.ChatMistralAI"),
        ):
            pipeline = EventRAGPipeline(fake_settings)

        with pytest.raises(ValueError, match="vide"):
            await pipeline.build_index([])

    # ======================= ``save_index()`` =======================

    # ---------------- Vérifie la persistence avec sauvegade sur disque si ``save_index()``
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_save_calls_save_local(self, rag_pipeline_with_index, tmp_path):
        """``save_index()`` doit invoquer ``FAISS.save_local`` avec le bon chemin."""
        rag_pipeline_with_index._index_path = tmp_path / "idx"
        rag_pipeline_with_index._vectorstore.save_local = MagicMock()

        await rag_pipeline_with_index.save_index()

        rag_pipeline_with_index._vectorstore.save_local.assert_called_once()

    # ---------------- Vérifie que ``save_index()`` lève une erreur si pas d'index
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_save_raises_when_no_index(self, fake_settings):
        """``save_index()`` doit lever une ``RuntimeError`` si aucun index n'existe."""
        with (
            patch("app.rag.rag_pipeline.MistralAIEmbeddings"),
            patch("app.rag.rag_pipeline.ChatMistralAI"),
        ):
            pipeline = EventRAGPipeline(fake_settings)

        with pytest.raises(RuntimeError, match="Pas d'index"):
            await pipeline.save_index()

    # ======================= ``load_index()`` =======================

    # ---------------- Vérifie que ``load_index()`` lève une erreur si pas d'index
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_load_returns_false_when_missing(self, fake_settings, tmp_path):
        """``load_index()`` doit retourner ``False`` quand les fichiers d'index sont absents."""
        fake_settings.faiss_index_path = tmp_path / "nonexistent"
        with (
            patch("app.rag.rag_pipeline.MistralAIEmbeddings"),
            patch("app.rag.rag_pipeline.ChatMistralAI"),
        ):
            pipeline = EventRAGPipeline(fake_settings)

        result = await pipeline.load_index()
        assert result is False
        assert pipeline.is_ready() is False

    # ---------------- Vérifie que le chargement ``load_index()`` a réussi si index
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_load_returns_true_when_index_exists(
        self, fake_settings, tmp_path, fake_embeddings
    ):
        """``load_index()`` doit retourner ``True`` quand les fichiers FAISS existent."""
        index_dir = tmp_path / "idx"
        index_dir.mkdir()
        # Touch the expected FAISS file so the existence check passes
        (index_dir / "index.faiss").touch()

        fake_settings.faiss_index_path = index_dir
        mock_vs = MagicMock()
        mock_vs.index.ntotal = 42
        mock_vs.as_retriever.return_value = MagicMock()

        with (
            patch("app.rag.rag_pipeline.MistralAIEmbeddings", return_value=fake_embeddings),
            patch("app.rag.rag_pipeline.ChatMistralAI"),
            patch("app.rag.rag_pipeline.FAISS.load_local", return_value=mock_vs),
        ):
            pipeline = EventRAGPipeline(fake_settings)
            result = await pipeline.load_index()

        assert result is True
        assert pipeline.is_ready() is True
        assert pipeline.document_count() == 42

    # ======================= ``query()`` ====================================

    # ---------------- Vérifie que ``query()`` ne marche pas si pipeline non initialisée
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_query_raises_when_not_ready(self, fake_settings):
        """Appeler ``query()`` sur un pipeline non initialisé doit lever une ``RuntimeError``."""
        with (
            patch("app.rag.rag_pipeline.MistralAIEmbeddings"),
            patch("app.rag.rag_pipeline.ChatMistralAI"),
        ):
            pipeline = EventRAGPipeline(fake_settings)

        with pytest.raises(RuntimeError, match="pas opérationnel"):
            await pipeline.query("Test ?")

    # ---------------- Vérifie les sortie de ``query()``
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_query_returns_expected_keys(
        self, rag_pipeline_with_index, sample_documents, fake_llm_response
    ):
        """Le résultat de ``query()`` doit contenir 'answer', 'source_documents' et 'scores'."""
        # Patch similarity_search_with_relevance_scores to avoid API calls
        rag_pipeline_with_index._vectorstore.similarity_search_with_relevance_scores = MagicMock(
            return_value=[(sample_documents[0], 0.92)]
        )
        # Patch the LCEL chain invoke
        rag_pipeline_with_index._chain = MagicMock()
        rag_pipeline_with_index._chain.invoke = MagicMock(return_value=fake_llm_response)

        result = await rag_pipeline_with_index.query("Où y a-t-il un concert de jazz ?", top_k=1)

        assert "answer" in result
        assert "source_documents" in result
        assert "scores" in result
        assert result["answer"] == fake_llm_response
        assert len(result["source_documents"]) == 1
        assert abs(result["scores"][0] - 0.92) < 1e-6

    # ---------------- Vérifie que answer de ``query()`` retourne bien une chaîne de caractère
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_query_answer_is_string(
        self, rag_pipeline_with_index, sample_documents, fake_llm_response
    ):
        """La réponse contenue dans le résultat doit être une chaîne de caractères non vide."""
        rag_pipeline_with_index._vectorstore.similarity_search_with_relevance_scores = MagicMock(
            return_value=[(sample_documents[0], 0.85)]
        )
        rag_pipeline_with_index._chain = MagicMock()
        rag_pipeline_with_index._chain.invoke = MagicMock(return_value=fake_llm_response)

        result = await rag_pipeline_with_index.query("Question de test ?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
