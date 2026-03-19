"""
tests/conftest.py

Script pour initier mocks et fixtures pytest pour l'ensemble de la suite de tests.


Toutes les entrées/sorties externes (API Mistral, API OpenAgenda, E/S disque FAISS)
sont remplacées par des bouchons (stubs) légers afin que les tests s'exécutent
hors-ligne, rapidement et de manière déterministe.
"""

# Imports
# Permet d'écrire des indices de type (ex: list[Document]) même si la classe n'est pas encore
# totalement définie, améliorant la compatibilité.
from __future__ import annotations

# import json
from pathlib import Path
from typing import Any  # , AsyncGenerator

# import pytest_asyncio # permet d'écrire des tests qui peuvent "attendre" (await) des réponses
# Client spécialisé qui simule des requêtes HTTP sur FastAPI sans devoir lancer un vrai serveur Web.
# from fastapi.testclient import TestClient
# patch: Remplace temporairement une classe ou une fonction par un mock.
# MagicMock: Un objet "caméléon" qui accepte n'importe quel appel de méthode.
# AsyncMock: La version asynchrone pour simuler, par exemple, une base de données ou un client HTTP.
from unittest.mock import MagicMock, patch  # , AsyncMock

# ------ Infrastructure RAG et Vectorielle -------
# bibliothèque optimisée pour recherche de similarité. avec wrapper LangChain pour interagir
# facilement avec l'index pour stocker les ``Document``
import faiss
import numpy as np

# ------------ Communication et systeme ----------
# Remplaçant moderne de ``requests``. Il gère l'asynchrone, ce qui est crucial pour
# OpenAgendaFetcher afin de ne pas bloquer tout le programme pendant qu'une page web charge.
# import httpx
# from httpx import AsyncClient
# ----------- Dépendances de base pour le testing ------------------
import pytest

# Dictionnaire en mémoire qui fait le lien entre ID d'un vecteur dans FAISS et ``Document``
from langchain_community.docstore.in_memory import InMemoryDocstore

# FAISS travaille pour l'essentiel avec des tableau NumPy type float32
from langchain_community.vectorstores import FAISS

# Unité de base de LangChain, l'objet comporte un ``page_content``et un ``metadata``
from langchain_core.documents import Document

# ---------- Résilience -------------
# Si l'API OpenAgenda subit une micro-coupure, Tenacity réessaiera automatiquement la requête
# selon une stratégie (ex: attendre 2s, puis 4s, puis 8s) avant d'abandonner.
# from tenacity import retry, stop_after_attempt, wait_exponential
# --------- Perso -------------
from app.etl.processor import EventDocumentProcessor
from app.rag.rag_pipeline import EventRAGPipeline

# =============================== DATAS =====================================


# SETTINGS
@pytest.fixture(scope="session")
def fake_settings():
    """
    Retourne un objet type ``Settings`` via ``SimpleNamespace`` avec des valeurs sûres.

    Utilise ``SimpleNamespace`` pour éviter de charger un vrai fichier ``.env``
    et déclencher la validation Pydantic qui exigerait de vraies clés API.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        # ======================== LLM =====================================
        mistral_base_url="https://api.mistral.ai/v1",
        mistral_api_key="test-mistral-key",
        llm_model="devstral-small-latest",
        llm_temperature=0.1,
        llm_max_tokens=1024,
        llm_top_p=0.9,
        embed_model="mistral-embed",
        chunk_size=200,
        chunk_overlap=50,
        # =================== OpenAgenda ===================================
        openagenda_public_url="test-oa-url",
        openagenda_updatedat="2025",
        openagenda_location_city="Paris",
        openagenda_location_region="Île-de-France",
        openagenda_limit=20,
        openagenda_offset=0,
        openagenda_lang="fr",
        openagenda_timezone="Europe/Berlin",
        openagenda_max_events=50,
        # ================== RAG ==========================================
        rag_top_k=3,
        faiss_index_path=Path("/tmp/test_faiss_index"),
        # ================ Securité /rebuild =======================================
        rebuild_api_key="test-rebuild-secret",
        # ================ Serveur ===============================================
        app_host="0.0.0.0",
        app_port=8000,
        app_env="development",
    )


# RAW_DATAS
@pytest.fixture(scope="session")
def raw_events() -> list[dict[str, Any]]:
    """
    Quelques exemples issu directement de OA avec la structure attendue (on a seulement modifié
    des "" et les null json en None pour correspondre au python)
    """
    return [
        {
            "uid": "43692627",
            "slug": "dedicace-de-peyo-lizarazu",
            "canonicalurl": "https://openagenda.com/librairie-mollat/events/dedicace-de-peyo-lizarazu",
            "title_fr": "Dédicace de Peyo Lizarazu",
            "description_fr": """Venez rencontrer Peyo Lizarazu lors d'une séance de dédicace de son
            livre 'Vies de surf' aux éditions La Martinière.""",
            "longdescription_fr": """<p>'Une découverte des vagues les plus majestueuses de la
            planète, accompagnés de portraits des personnalités marquantes du monde du surf,
            de documents d'archives et de photographies.' ©Electre 2022</p> <p></p> <p>Rendez-vous
            dès 15 heures à la Librairie Mollat.</p>""",
            "conditions_fr": None,
            "keywords_fr": None,
            "image": """
            https://cibul.s3.amazonaws.com/d0962598cf834ec58e32a23ccc3422a5.base.image.jpg""",
            "imagecredits": None,
            "thumbnail": """
            https://cibul.s3.amazonaws.com/d0962598cf834ec58e32a23ccc3422a5.thumb.image.jpg""",
            "originalimage": """
            https://cibul.s3.amazonaws.com/d0962598cf834ec58e32a23ccc3422a5.full.image.jpg""",
            "updatedat": "2022-10-25T09:00:46+00:00",
            "daterange_fr": "Mercredi 16 novembre, 15h00",
            "firstdate_begin": "2022-11-16T14:00:00+00:00",
            "firstdate_end": "2022-11-16T17:00:00+00:00",
            "lastdate_begin": "2022-11-16T14:00:00+00:00",
            "lastdate_end": "2022-11-16T17:00:00+00:00",
            "timings": """
            [{'begin': '2022-11-16T15:00:00+01:00', 'end': '2022-11-16T18:00:00+01:00'}]""",
            "accessibility": None,
            "accessibility_label_fr": None,
            "location_uid": "65949775",
            "location_coordinates": {"lon": -0.578647, "lat": 44.840868},
            "location_name": "Librairie Mollat",
            "location_address": "15 rue Vital Carles, 33000 Bordeaux",
            "location_district": "Triangle d'Or",
            "location_insee": "33063",
            "location_postalcode": "33000",
            "location_city": "Bordeaux",
            "location_department": "Gironde",
            "location_region": "Nouvelle-Aquitaine",
            "location_countrycode": "FR",
            "location_image": None,
            "location_imagecredits": None,
            "location_phone": None,
            "location_website": None,
            "location_links": None,
            "location_tags": None,
            "location_description_fr": None,
            "location_access_fr": None,
            "attendancemode": """
            {'id': 1, 'label': {'fr': 'Sur place', 'en': 'Offline', 'it': 'In presenza',
            'es': 'Desconnectad', 'de': 'Offline', 'br': 'War al lec\u2019h',
            'io': 'crwdns14266:0crwdne14266:0'}}""",
            "onlineaccesslink": None,
            "status": """
            {'id': 1, 'label': {'fr': 'Programm\u00e9', 'en': 'Scheduled',
            'io': 'crwdns16100:0crwdne16100:0'}}""",
            "age_min": None,
            "age_max": None,
            "originagenda_title": "Librairie Mollat",
            "originagenda_uid": "30224219",
            "contributor_email": None,
            "contributor_contactnumber": None,
            "contributor_contactname": None,
            "contributor_contactposition": None,
            "contributor_organization": None,
            "category": None,
            "country_fr": "France (Métropole)",
            "registration": None,
            "links": None,
        },
        {
            "uid": "61763667",
            "slug": "la-peur-en-voyage",
            "canonicalurl": """
            https://openagenda.com/reseau-des-mediatheques-du-mans/events/la-peur-en-voyage""",
            "title_fr": "La peur en voyage",
            "description_fr": """Venez frissonner à l’international ! Lectures réalisées en
            partenariat avec l’association AFaLaC.""",
            "longdescription_fr": """<p>Venez frissonner à l’international !<br />Lectures réalisées
            en partenariat avec l’association AFaLaC.<br /><em>À partir de 4 ans.</em></p>""",
            "conditions_fr": None,
            "keywords_fr": None,
            "image": """
            https://cibul.s3.amazonaws.com/fe5e41c82f1242268a426d8f2820e36e.base.image.jpg""",
            "imagecredits": None,
            "thumbnail": """
            https://cibul.s3.amazonaws.com/fe5e41c82f1242268a426d8f2820e36e.thumb.image.jpg""",
            "originalimage": """
            https://cibul.s3.amazonaws.com/fe5e41c82f1242268a426d8f2820e36e.full.image.jpg""",
            "updatedat": "2023-01-03T16:01:24+00:00",
            "daterange_fr": "Samedi 21 janvier, 19h00",
            "firstdate_begin": "2023-01-21T18:00:00+00:00",
            "firstdate_end": "2023-01-21T18:30:00+00:00",
            "lastdate_begin": "2023-01-21T18:00:00+00:00",
            "lastdate_end": "2023-01-21T18:30:00+00:00",
            "timings": """
            [{'begin': '2023-01-21T19:00:00+01:00', 'end': '2023-01-21T19:30:00+01:00'}]""",
            "accessibility": None,
            "accessibility_label_fr": None,
            "location_uid": "87034922",
            "location_coordinates": {"lon": 0.192803, "lat": 48.00265},
            "location_name": "Médiathèque Aragon",
            "location_address": "54 rue du Port, 72 000 Le Mans",
            "location_district": None,
            "location_insee": "72181",
            "location_postalcode": "72000",
            "location_city": "Le Mans",
            "location_department": "Sarthe",
            "location_region": "Pays de la Loire",
            "location_countrycode": "FR",
            "location_image": None,
            "location_imagecredits": None,
            "location_phone": None,
            "location_website": None,
            "location_links": None,
            "location_tags": None,
            "location_description_fr": None,
            "location_access_fr": None,
            "attendancemode": """
            {'id': 1, 'label': {'fr': 'Sur place', 'en': 'Offline', 'it': 'In presenza',
            'es': 'Desconnectad', 'de': 'Offline', 'br': 'War al lec\u2019h',
            'io': 'crwdns14266:0crwdne14266:0'}}""",
            "onlineaccesslink": None,
            "status": """{'id': 1, 'label': {'fr': 'Programm\u00e9', 'en': 'Scheduled',
            'io': 'crwdns16100:0crwdne16100:0'}}""",
            "age_min": 4,
            "age_max": 99,
            "originagenda_title": "Réseau des médiathèques du Mans",
            "originagenda_uid": "48454528",
            "contributor_email": None,
            "contributor_contactnumber": None,
            "contributor_contactname": None,
            "contributor_contactposition": None,
            "contributor_organization": None,
            "category": None,
            "country_fr": "France (Métropole)",
            "registration": None,
            "links": None,
        },
    ]


# # Document LangChain
# @pytest.fixture(scope="session")
# def sample_documents() -> list[Document]:
#     """
#     Objets ``Document`` LangChain issu de la fixture ``raw_events`` pour tester si le document est
#     correctement produit.
#     """
#     return [
#         Document(
#             page_content=(
#                 "Titre : Dédicace de Peyo Lizarazu\n"
#                 "Détails : Une découverte des vagues les plus majestueuses de la planète, "
#                 "accompagnés de portraits des personnalités marquantes du monde du surf... "
#                 "Rendez-vous dès 15 heures à la Librairie Mollat.\n"
#                 "Adresse : 15 rue Vital Carles, 33000 Bordeaux\n"
#                 "Ville : Bordeaux\n"
#                 "Téléphone : \n"
#                 "Web : \n"
#                 "Acces : \n"
#                 "handicap : accessibilité réduite\n"
#                 "Prix : "
#             ),
#             metadata={
#                 "uid": "43692627",
#                 "titre": "Dédicace de Peyo Lizarazu",
#                 "city": "Bordeaux",
#                 "venue": "Librairie Mollat",
#                 "age_min": None,
#                 "price": None,
#                 "date_start": "2022-11-16T14:00:00+00:00",
#                 "url": "https://openagenda.com/librairie-mollat/events/dedicace-de-peyo-lizarazu"
#             },
#         ),
#         Document(
#             page_content=(
#                 "Titre : La peur en voyage\n"
#                 """Détails : Venez frissonner à l’international ! Lectures réalisées en
#                   partenariat avec l’association AFaLaC. À partir de 4 ans.\n"""
#                 "Adresse : 54 rue du Port, 72 000 Le Mans\n"
#                 "Ville : Le Mans\n"
#                 "Téléphone : \n"
#                 "Web : \n"
#                 "Acces : \n"
#                 "handicap : accessibilité réduite\n"
#                 "Prix : "
#             ),
#             metadata={
#                 "uid": "61763667",
#                 "titre": "La peur en voyage",
#                 "city": "Le Mans",
#                 "venue": "Médiathèque Aragon",
#                 "age_min": 4,
#                 "price": None,
#                 "date_start": "2023-01-21T18:00:00+00:00",
#                 "url": """
#                 https://openagenda.com/reseau-des-mediatheques-du-mans/events/la-peur-en-voyage"""
#             },
#         ),
#     ]
#  Ici, on ne teste que si le RAG comprends ce que la méthdoe processor lui envois plutot que le
#  document processé en lui-même
@pytest.fixture(scope="session")
def sample_documents(raw_events, fake_settings) -> list[Document]:
    """
    Génère dynamiquement les objets ``Document`` en utilisant le processeur réel.
    Cela garantit que les tests de RAG utilisent exactement le format produit
    par l'ETL, sans double saisie manuelle.
    """

    # On initialise le processeur avec nos réglages de test
    processor = EventDocumentProcessor(fake_settings)

    # On transforme les données brutes en documents LangChain
    documents = processor.process(raw_events)

    return documents


# Réponse LLM sur le premier document
@pytest.fixture(scope="session")
# def fake_llm_response() -> str:
#     """Réponse LLM pré-enregistrée."""
#     return (
#         "Peyo Lizarazu sera présent pour une séance de dédicace de son livre 'Vies de surf' "
#         "le mercredi 16 novembre à partir de 15h. L'événement se déroule à la "
#         "Librairie Mollat, située au 15 rue Vital Carles à Bordeaux."
#     )
def fake_llm_response() -> str:
    return "..."


# =================================================================================================


class _FakeEmbeddings:
    """
    Embeddings déterministes renvoyant des vecteurs aléatoires de taille fixe.\n
    Entièrement hors-ligne — aucun appel API. Utilise un générateur (RNG) avec
    une graine (seed) fixe pour que les distances entre vecteurs soient
    reproductibles d'un test à l'autre.
    """

    def __init__(self, size: int = 384) -> None:
        self._size = size
        self._rng = np.random.default_rng(seed=42)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Retourne un vecteur unitaire aléatoire par texte."""
        # On s'assure que c'est du float32 pour FAISS
        vecs = self._rng.random((len(texts), self._size)).astype(np.float32)
        # L2-normalise so cosine similarity == dot product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / norms).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Retourne un seul vecteur unitaire aléatoire."""
        return self.embed_documents([text])[0]


@pytest.fixture(scope="session")
def fake_embeddings() -> _FakeEmbeddings:
    """Instance d'embeddings factices (hors-ligne, sans appels API)."""
    return _FakeEmbeddings(size=384)


# ── RAG pipeline fixture ───────────────────────────────────────────────────────


# Pas besoin de pytest_asyncio.fixture ici si on ne fait pas d'await
@pytest.fixture
def rag_pipeline_with_index(fake_settings, sample_documents, fake_embeddings):
    """
    Retourne un ``EventRAGPipeline`` avec un index FAISS pré-rempli via
    ``sample_documents`` et des embeddings factices — aucun appel API Mistral.
    """

    # Construction d'un index FAISS réel (minuscule) avec embeddings factices
    vecs = np.array(
        fake_embeddings.embed_documents([doc.page_content for doc in sample_documents]),
        dtype=np.float32,
    )
    dim = vecs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)  # type: ignore

    # Patch des classes MistralAI avant l'instanciation
    with (
        patch("app.rag.rag_pipeline.MistralAIEmbeddings", return_value=fake_embeddings),
        patch("app.rag.rag_pipeline.ChatMistralAI"),
        # patch("CLI.eval_rag.AsyncOpenAI"), # Patch du bridge OpenAI
        # patch("CLI.eval_rag.llm_factory"),
        # patch("CLI.eval_rag.embedding_factory", return_value=fake_embeddings),
    ):
        pipeline = EventRAGPipeline(fake_settings)

    # # Création du vectorstore LangChain
    # Injection directe du vectorstore (évite les calculs d'indexation réels)
    docstore_dict = {str(i): doc for i, doc in enumerate(sample_documents)}
    index_to_docstore_id = {i: str(i) for i in range(len(sample_documents))}
    pipeline._vectorstore = FAISS(
        embedding_function=fake_embeddings,
        index=index,
        docstore=InMemoryDocstore(docstore_dict),
        index_to_docstore_id=index_to_docstore_id,
    )
    pipeline._doc_count = len(sample_documents)
    pipeline._chain = MagicMock()  # Sera remplacé dans les tests unitaires
    return pipeline


# ── FastAPI test client fixture ────────────────────────────────────────────────

# @pytest.fixture
# def test_app(fake_settings, rag_pipeline_with_index):
#     """
#     Return a FastAPI ``TestClient`` with the RAG pipeline pre-loaded and
#     all external dependencies patched.
#     """
#     from app.main import create_app

#     application = create_app()
#     # Bypass lifespan by injecting state directly
#     application.state.settings = fake_settings
#     application.state.rag = rag_pipeline_with_index

#     with TestClient(application, raise_server_exceptions=True) as client:
#         yield client
