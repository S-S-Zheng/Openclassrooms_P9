"""
scripts/build_index.py

Script CLI autonome qui récupère les événements OpenAgenda, les traite,
construit et sauvegarde l'index FAISS sans démarrer le serveur API.\n
Ce script peut également être utilisé lors de la construction de l'image Docker 
pour pré-charger l'index.
"""

# imports
import asyncio
import logging
import sys
import time
from pathlib import Path

# S'assurer que la racine du projet est dans le sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.config import get_settings
from app.data.fetcher import OpenAgendaFetcher
from app.data.processor import EventDocumentProcessor
from app.rag.rag_pipeline import EventRAGPipeline

# Configuration du logger pour l'affichage console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===============================================================


async def indexation() -> None:
    """
    Orchestre la pipeline complete de construction de l'index.

    Pipeline ETL (Extract, Transform, Load)
    ------
    Charge config -> Fetch raw data -> Preproc en Document LangChain

    Pipeline Index
    ----------
    Embedding -> pesistence
    """
    # ========================= Initialisation =========================
    settings = get_settings()
    # C'est une horloge qui ne revient jamais en arrière, même si le système met à jour
    # l'heure via internet (NTP). On l'utilise pour mesurer des durées précises de traitement.
    # time.time() peut fluctuer, monotonic() est stable.
    start = time.monotonic()
    # Instanciation
    rag = EventRAGPipeline(settings)

    # # ================== Chargement d'index impossible ===========================
    # if not await rag.load_index():
    # logger.info("Index absent. Lancement du process ETL...")
    logger.info("=== Indexation ... ===")
    logger.info(f"Récupération des événements pour {settings.openagenda_location_city}...")
    logger.info(f"Max événements: {settings.openagenda_max_events}")
    logger.info(f"Modèle d'embedding: {settings.embed_model}")
    logger.info(f"Chemin d'index: {settings.faiss_index_path}")

    # -------------------- Fetcher (Raw data) --------------------
    fetcher = OpenAgendaFetcher(settings)
    logger.info("Récupération des événements ...")
    raw_events = await fetcher.fetch_events()
    logger.info(f"{len(raw_events)} événements bruts récupérés.")

    if not raw_events:
        logger.error("Aucun événement récupéré. Vérifiez l'URL, la clé API ou les quotas.")
        sys.exit(1)

    # -------------------- Processor (Nettoyage et Chunking) --------------------
    processor = EventDocumentProcessor(settings)
    documents = processor.process(raw_events)
    logger.info(f"{len(documents)} documents prêts pour l'indexation.")

    # -------------------- Embedding (Indexation et persistence) --------------------
    logger.info("Construction de l'index FAISS (Coffee time)...")
    await rag.build_index(documents)
    await rag.save_index()

    duration = time.monotonic() - start
    logger.info(f"=== Indexation terminée en {duration:.2f} s ===")
    logger.info(f"Nb de documents indexés: {rag.document_count()}")
    logger.info(f"Chemin d'index: {settings.faiss_index_path}")

    # # ================== Chargement d'index réussi =========================== 
    # else:
    #     logger.info("Index chargé depuis le disque.")
    #     duration = time.monotonic() - start
    #     logger.info(f"=== Chargement terminée en {duration:.2f} s ===")

# ================================================================


if __name__ == "__main__":
    try:
        asyncio.run(indexation())
    except KeyboardInterrupt:
        logger.warning("\nInterruption par l'utilisateur.")
        sys.exit(0)
