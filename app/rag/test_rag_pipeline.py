"""
Script pour test tout fonctionne du chargement au processing (bonus embedding et vectore store).

Pipeline RAG:\n
Charge config -> Fetch raw data -> Preproc en Document LangChain -> Embedding + Vector store.

Test:\n
Question (query) -> check Vector store (similarité) -> Interprétation LLM (absent ici) -> Réponse.
"""

# imports
import asyncio

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings

# ROOT_PATH = Path(__file__).resolve().parents[2]
from app.core.config import get_settings
from app.data.fetcher import OpenAgendaFetcher
from app.data.processor import EventDocumentProcessor

# =========================================


async def test_rag():
    # -------------------- Charger la configuration --------------------
    # Bien que get_settings serve de pilier centrale aux vars, il faut charger .env afin qu'il
    # ne passe pas que par les valeurs par défaut.
    load_dotenv()
    settings = get_settings()
    print("--- Configuration chargée ---")

    # -------------------- Fetcher (Raw data) --------------------
    fetcher = OpenAgendaFetcher(settings)
    print(f"Récupération des événements pour {settings.openagenda_location_city}...")
    raw_events = await fetcher.fetch_events()
    print(f"{len(raw_events)} événements bruts récupérés.")

    # # Sauvegarde le JSON brut pour inspection
    # import json
    # with open("debug_raw_events.json", "w", encoding="utf-8") as f:
    #     json.dump(raw_events, f, indent=4, ensure_ascii=False)
    # print(f"Fichier debug_raw_events.json créé avec {len(raw_events)} événements.")

    # -------------------- Processor (Nettoyage et Chunking) --------------------
    processor = EventDocumentProcessor(settings)
    documents = processor.process(raw_events)
    print(f"{len(documents)} documents prêts pour l'indexation.")

    # -------------------- Embedding (Indexation) --------------------
    print("Vectorisation en cours avec Mistral (prend un café)...")
    embeddings = MistralAIEmbeddings(
        mistral_api_key=settings.mistral_api_key,  # type:ignore
        model=settings.embed_model,
    )
    # Création de la base FAISS en mémoire
    vector_store = FAISS.from_documents(documents, embeddings)
    print("Index FAISS créé avec succès !")

    # =========================== TEST ===========================
    query = "Existe-t-il des activités pour enfants à Paris ?"
    print(f"\n--- Test de recherche pour : '{query}' ---")

    # Recherche de similarité
    results = vector_store.similarity_search(query, k=3)

    for i, res in enumerate(results):
        print(f"\nRésultat #{i + 1}:")
        print(res.page_content)
        print(f"Source : {res.metadata.get('url')}")

    # -------------------- Save INDEX localement pour ré indexage --------------------
    vector_store.save_local("faiss_index_test")
    print("\nIndex sauvegardé dans le dossier 'faiss_index_test'.")


# =======================================================


if __name__ == "__main__":
    asyncio.run(test_rag())
