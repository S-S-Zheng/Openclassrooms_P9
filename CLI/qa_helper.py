"""
CLI/qa_helper.py

Script CLI permettant de fouiller l'index est de ressortir titre, uid et un extrait de quelque
events afin de pouvoir rédiger le QA d'évaluation sans faire de data leakage ou choux blanc.\n
On randomize le picking des event (pas vraiment nécéssaire) de façon deterministe (le seed 42).
"""
# imports
import asyncio
import sys
from pathlib import Path
import random


# Ajout du dossier racine au path pour permettre les imports relatifs
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from app.rag.rag_pipeline import EventRAGPipeline
from app.core.config import get_settings


# ==============================================================


async def inspect_index():
    settings = get_settings()
    rag = EventRAGPipeline(settings)
    await rag.load_index()

    # Extraction via l'attribut docstore (Méthode robuste)
    # Dans les versions récentes de LangChain, on peut accéder directement aux documents
    docstore = rag._vectorstore.docstore # type:ignore
    
    # Si c'est un InMemoryDocstore, il possède un attribut _dict
    if hasattr(docstore, "_dict"):
        all_docs = list(docstore._dict.values()) # type:ignore
    else:
        # Fallback sur les IDs
        all_ids = list(rag._vectorstore.index_to_docstore_id.values()) # type:ignore
        all_docs = [docstore.search(doc_id) for doc_id in all_ids]

    if not all_docs:
        print("L'index est chargé mais ne contient aucun document.")
        return

    print("-" * 50)
    # random deterministe
    random.seed(42)
    random_docs = random.sample(all_docs, 5)

    # Affichage de N documents random pour créer ensuite la question et le ground truth
    for i, doc in enumerate(random_docs):
        title = doc.metadata.get("title") # type:ignore
        uid = doc.metadata.get("uid") # type:ignore
        print(f"[{i+1}] {title} (ID: {uid})")
        # On affiche un petit bout du contenu pour vérifier le texte
        extract = doc.page_content[:400].replace('\n', ' ') # type:ignore
        print(f"    Extrait: {extract}...")
        print("-" * 50)


# =====================================================


if __name__ == "__main__": #pragma: no cover
    try:
        asyncio.run(inspect_index())
    except Exception as exc:
        print(f"Le script a crashé : {exc}")