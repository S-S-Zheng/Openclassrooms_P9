"""
app/rag/pipeline.py

Système RAG basé sur BaseRAG utilisant:
- MistralAI Embeddings: Pour l'indexation
- FAISS(CPU): La base de donnée vectorielle
- ChatMistralAI: Le LLM qui lira le contexte (search and Retrieve) et rédigera la réponse
- LangChain LCEL: Le connecteur de chaque étape. Utilise l'operateur ``|`` (pipe).

FAISS et les appel LLM sont bloquants i.e tout le serveur FastAPI bloque toute interaction
(avec tout le monde) le temps de la recherche. Pour éviter cela, on délègue ces tâches lourdes
à des pool de threads isolés via ``asyncio.to_thread`` afin que le serveur reste fonctionnel.
"""

# Imports
import asyncio
import logging
from pathlib import Path
from typing import Any

# FAISS : Moteur de recherche vectorielle (ANN). Stocke les vecteurs et calcule les distances.
from langchain_community.vectorstores import FAISS

# Structure de donnée standard de LangChain (Texte + Metadata).
from langchain_core.documents import Document

# Convertit la réponse complexe du LLM en une simple chaîne de caractères.
from langchain_core.output_parsers import StrOutputParser

# Gère la fusion du texte du fichier .txt avec les variables {context} et {question}.
from langchain_core.prompts import ChatPromptTemplate

# Permet de faire passer la question utilisateur inchangée à travers la chaîne.
# Embeddings pour transformer le texte en nombres, Chat pour l'intelligence.
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from app.core.config import Settings
from app.rag.base_rag import BaseRAG

logger = logging.getLogger(__name__)

PROMPT_MANAG_FILE = Path(__file__).parent / "prompt_management.txt"

# ============================================================================


class EventRAGPipeline(BaseRAG):
    """
    RAG pipeline.

    Parameters
    ----------
    settings:
        Paramètres (API, URL, configuration...).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._index_path: Path = settings.faiss_index_path
        # Chargement du prompt management
        with open(PROMPT_MANAG_FILE, "r", encoding="utf-8") as f:
            self._prompt_text = f.read()
        # Initie les modèles UNE FOIS; ils sont STATELESS et THREAD-SAFE
        self._embeddings = MistralAIEmbeddings(
            model=settings.embed_model,
            mistral_api_key=settings.mistral_api_key,  # type:ignore
        )
        self._llm = ChatMistralAI(
            model_name=settings.llm_model,
            mistral_api_key=settings.mistral_api_key,  # type:ignore
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            top_p=settings.llm_top_p,
        )

        self._vectorstore: FAISS | None = None
        self._chain: Any | None = None
        self._doc_count: int = 0

    async def build_index(self, documents: list[Document]) -> None:
        """
        Vectorise documents et construit un index FAISS en background via asyncio.to_thread.

        Parameters
        ----------
        documents (list[Document])
            Objet ``Document`` pré-traité et prêt pour indexation.

        Raises
        ------
        ValueError
            Si ``documents`` est vide.
        """
        if not documents:
            raise ValueError("Liste de documents vide.")

        logger.info(f"Indexation FAISS à partir de {len(documents)} documents …")

        # Création de la base FAISS en mémoire. FAISS.from_documents est une fonction synchrone qui
        # attend la réponse du réseau (Mistral Embed)
        self._vectorstore = await asyncio.to_thread(
            FAISS.from_documents, documents, self._embeddings
        )
        self._doc_count = len(documents)
        self._chain = self._build_chain()  # Prépare la chaîne par défaut
        logger.info(f"Index FAISS créé avec succès ! ({self._doc_count} documents)")

    async def save_index(self) -> None:
        """
        Sauvegarde l'index courant dans le chemin soumis dans ``settings.faiss_index_path``.\n
        Créé le chemin s'il n'existe pas.
        """
        if self._vectorstore is None:
            raise RuntimeError("Pas d'index à save, appeler build_index() d'abord.")

        self._index_path.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._vectorstore.save_local, str(self._index_path))
        logger.info(f"Index sauvegardé dans le dossier {self._index_path}")

    async def load_index(self, index_filename: str = "index.faiss") -> bool:
        """
        Charge un index FAISS depuis le disque.\n
        Le checking vérifie la présence physique du fichier .faiss et .pkl (métadonnées).

        Parameters
        ----------
        index_filename (str)
            Nom du fichier index. Par défaut ``index.faiss``

        Returns
        -------
        bool
            ``True`` Si l'index existe et a été chargé correctement\n
            ``False`` Si pas d'index trouvé
        """
        index_file = self._index_path / index_filename
        if not index_file.exists():
            logger.warning(f"Aucun index trouvé dans {self._index_path}")
            return False

        logger.info(f"Chargement de {index_file} …")
        # Reconnecte l'index chargé à l'objet self._embeddings pour que les futures questions
        # soient converties avec le même modèle mathématique.
        self._vectorstore = await asyncio.to_thread(
            FAISS.load_local,
            str(self._index_path),
            self._embeddings,
            allow_dangerous_deserialization=True,  # requis pour LangChain ≥ 0.2 pour load pickle
        )

        # Compte le nombre de doc total indéxé
        self._doc_count = self._vectorstore.index.ntotal
        self._chain = self._build_chain()
        logger.info(f"Index chargé ({self._doc_count} vecteurs)")
        return True

    async def query(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """
        Lance la requete dans le système RAG.\n
        Les étapes de Récupération et de LLM sont sous traitées à des pool de threads isolés
        pour éviter de bloquer FastAPI.

        Parameters
        ----------
        question:
            Question utilisateur
        top_k:
            Nb de source de doc à récupérer. On utilise le config de LangChain pour passer le
            paramètre 'k' au retriever SANS reconstruire l'objet chain.
            C'est thread-safe et performant.

        Returns
        -------
        dict[str, Any]
            Keys: ``"answer"``, ``"source_documents"``, ``"scores"``.

        Raises
        ------
        RuntimeError
            Si l'index n'a pas encore été construit ou chargé.
        """
        if not self.is_ready():
            raise RuntimeError("Le système RAG n'est pas opérationnel")

        # --------------------------- RETRIEVE STEP ---------------------------
        # On utilise une valeur k dynamique (priorité : argument > config > défaut)
        k_value = top_k or self._settings.rag_top_k

        # Récupération des docs source avec score de similarité
        if self._vectorstore is not None:  # Seulement pour satisfaire Pylance
            retriever_fn = self._vectorstore.similarity_search_with_relevance_scores
        results: list[tuple[Document, float]] = await asyncio.to_thread(
            retriever_fn, question, k=k_value
        )
        source_docs: list[Document] = [doc for doc, _ in results]
        scores: list[float] = [float(score) for _, score in results]

        # --------------------------- AUGMENTATED and GENERATION STEPS ---------------------------
        # Génère la réponse au travers du LCEL
        # ----------------------------------
        # chain = (
        #     self._build_chain(k=top_k)
        #     if top_k != self._settings.rag_top_k
        #     else self._chain
        # )
        # if chain is not None: # Satisfaire Pylance
        #     answer: str = await asyncio.to_thread(
        #         chain.invoke,
        #         question
        #     )
        # ----------------------------------
        # ----------------------------------
        # On injecte k_value dynamiquement dans la configuration de l'invocation
        # if self._chain is not None: # Satisfaire Pylance
        #     answer = await asyncio.to_thread(
        #         self._chain.invoke,
        #         question,
        #         config={"configurable": {"search_kwargs": {"k": k_value}}}
        #     )
        # ----------------------------------
        # On appelle la chaîne en lui injectant DIRECTEMENT les docs déjà trouvés
        # Au lieu de lui passer 'question' (str), on lui passe un dico avec le contexte déjà prêt
        if self._chain is not None:
            answer = await asyncio.to_thread(
                self._chain.invoke,
                {"question": question, "docs": source_docs},  # On passe les docs ici
            )

        return {
            "answer": answer,
            "source_documents": source_docs,
            "scores": scores,
        }

    # ======================== HELPERS ===========================================

    def _build_chain(self) -> Any:
        """
        Emploie l'opérateur '|' (LCEL) :
        - Le dictionnaire prépare les entrées.
        - 'retriever | format_docs' : Récupère les docs et les transforme en bloc de texte.
        - 'prompt' : Fusionne le tout.
        - 'llm' : Génère la réponse brute.
        - 'parser' : Extrait uniquement le texte de la réponse.

        Returns
        -------
        Runnable
            LangChain LCEL ``Runnable`` prêt à être invoquer avec une question.
        """
        # k et retriever plus nécéssaire car on fait un seul appel a la base et on emploi
        # directement la doc issue de FAISS. La chaîne est alors plus rapide et efficace.
        # ---------------------------------------
        # effective_k = k or self._settings.rag_top_k
        # if self._vectorstore is not None: # Satisfaction Pylance
        #     retriever = self._vectorstore.as_retriever(
        #         search_type="similarity",
        #         # search_kwargs={"k": effective_k},
        #         search_kwargs={"k": self._settings.rag_top_k}
        #     )
        # ---------------------------------------
        prompt = ChatPromptTemplate.from_template(self._prompt_text)

        # LCEL pipeline
        chain = (
            # {
            #     "context": retriever | self._format_docs, #type: ignore
            #     "question": RunnablePassthrough(),
            # }
            {
                # On récupère les docs déjà trouvés et on les formate
                "context": lambda x: self._format_docs(x["docs"]),
                "question": lambda x: x["question"],
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )
        return chain

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        """
        Concatène le contenu des documents dans une str unique pour le prompt.

        Parameters
        ----------
        docs:
            Objets ``Document`` à récup'

        Returns
        -------
        str
            La châine de caractère issu des docs compactés avec double saut de ligne.
        """
        sections = [
            f"\nRésultat #{i + 1}\n{doc.page_content}\nSource: {doc.metadata.get('url')}"
            for i, doc in enumerate(docs)
        ]
        return "\n\n".join(sections)

    def is_ready(self) -> bool:
        """
        Permet de s'assurer que les requêtes peuvent se lancer en checkant que le vector store et
        les connecteurs LCEL ne sont pas vide. Renvoi ``True`` si ok.
        """
        return self._vectorstore is not None and self._chain is not None

    def document_count(self) -> int:
        """
        Renvoi le nb de doc actuellement stocké dans l'index (0 si vide ou non initialisé).
        """
        return self._doc_count
