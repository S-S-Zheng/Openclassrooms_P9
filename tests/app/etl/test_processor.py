"""
tests/app/etl/test_processor.py
──────────────────────
Unit tests for the etl pipeline:

* ``EventDocumentProcessor`` — validates text extraction and Document creation.

All HTTP calls are mocked via ``pytest-mock`` and ``respx`` / ``unittest.mock``
so no real API keys are required.
"""

# imports
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.etl.processor import EventDocumentProcessor

# =========================================================


# On applique le marqueur 'unit' à toute la classe
@pytest.mark.unit
class TestEventDocumentProcessor:
    # ------------------- Instancie l'objet de pré traitement avec desdonnées fake
    @pytest.fixture
    def processor(self, fake_settings) -> EventDocumentProcessor:
        """Instance du processeur avec les réglages de test."""
        return EventDocumentProcessor(fake_settings)

    # ------------------ Vérifie que process() renvoi un doc langchain
    def test_process_returns_documents(self, processor, raw_events):
        """Vérifie que chaque ``process()`` produit au moins un ``Document`` par event."""
        docs = processor.process(raw_events)

        # Assertions
        assert len(docs) >= len(raw_events)
        assert all(isinstance(doc, Document) for doc in docs)

    # ------------------ Vérifie la présence obligatoire du titre dans le texte
    def test_document_page_content_contains_title(self, processor, raw_events):
        """Le titre de l'événement doit être présent dans le texte final."""
        docs = processor.process(raw_events)
        # Premier event de conftest: "Dédicace de Peyo Lizarazu"
        content = [doc.page_content for doc in docs]

        # Assertions
        assert any("Peyo Lizarazu" in cont for cont in content)

    # ------------------  Vérifie les keys dans metadata_fields
    def test_document_metadata_keys(self, processor, raw_events):
        """Vérifie la présence des clés de métadonnées essentielles."""
        required_keys = {"uid", "title", "city", "url", "date_start"}
        docs = processor.process(raw_events)

        # Assertions
        for doc in docs:
            assert required_keys.issubset(doc.metadata.keys()), (
                f"Clés manquantes dans : {doc.metadata.keys()}"
            )

    # ------------------ Vérifie la robstesse du processor en cas de liste vide
    def test_process_empty_list(self, processor):
        """Une liste vide ne doit pas faire planter le processeur."""
        docs = processor.process([])

        # Assertions
        assert docs == []

    # -------------- Vérifie la robustesse en cas d'event vide, non dict ou sans clef obligatoire
    def test_process_skips_malformed_events(self, processor):
        """Les événements corrompus ou vides doivent être ignorés proprement."""
        malformed = [None, {}, {"uid": ""}]
        # On s'attend à 0 documents car les champs requis (titre, ville) manquent
        docs = processor.process(malformed)  # type: ignore

        # Assertions
        assert len(docs) == 0

    # ------------------ Vérifie si le chunking fonctionne correctement
    def test_long_description_is_chunked(self, processor, fake_settings):
        """Vérifie que le découpage (chunking) fonctionne pour les textes longs."""
        # On réduit la taille des chunks pour forcer le split facilement
        fake_settings.chunk_size = 100
        fake_settings.chunk_overlap = 10

        long_text = """<p>'Une découverte des vagues les plus majestueuses de la planète,
        accompagnés de portraits des personnalités marquantes du monde du surf, de documents
        d'archives et de photographies.' ©Electre 2022</p> <p></p> <p>Rendez-vous dès 15 heures
        à la Librairie Mollat.</p>"""
        event = {
            "uid": "999",
            "title_fr": "Événement Long",
            "longdescription_fr": long_text,
            "location_city": "Paris",
            "location_address": "1 rue de Paris",
            "firstdate_begin": "2025-01-01T10:00:00Z",
        }
        docs = processor.process([event])

        # Assertions
        # Si le texte fait approx 2000 chars et le chunk 100, on doit avoir plusieurs docs
        assert len(docs) > 1

    # ------------------ Vérifie si le chunking fonctionne correctement
    def test_process_short_event_no_chunking(self, processor):
        """Vérifie qu'un événement court est ajouté sans être découpé."""
        # On force une taille de chunk suffisante (100 approx 75 mots)
        processor._chunk_size = 500

        short_text = "<p>Rendez-vous dès 15 heures à la Librairie Mollat.</p>"
        event = {
            "uid": "999",
            "title_fr": "Événement court",
            "longdescription_fr": short_text,
        }
        # Le texte produit sera court, donc on passera dans le 'else: documents.append(doc)'
        docs = processor.process([event])

        # Assertions
        assert len(docs) == 1

    # ------------------ Vérifie que la donnée est bien extrait de la métadonnée
    def test_metadata_extraction_consistency(self, processor, raw_events):
        """Vérifie que la ville est correctement extraite des données brutes."""
        docs = processor.process([raw_events[0]])  # Bordeaux
        assert docs[0].metadata["city"] == "Bordeaux"

        docs_mans = processor.process([raw_events[1]])  # Le Mans
        assert docs_mans[0].metadata["city"] == "Le Mans"
