"""
app/rag/base_rag.py

Abstract base class (ABC) qui définie le contrat que toutes les classes héritières RAG devront
respecter.

Le système RAG doit être agnostique (ne considère pas de forme particulière de source) et n'attend
que des objets Documents standardisés.\n
Il gère trois rôles:
    - Le cycle de vie du système qui se concentre sur l'indexation et sa gestion long terme
    - Les requêtes/réponses
    - Le monitoring du sytème afin de s'assurer que la pipeline est fonctionnelle
"""

# imports
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document

# =====================================================================


class BaseRAG(ABC):
    """
    ABC pour système RAG.\n
    Toutes classe héritière doit obligatoirement implémenter les abstractmethod qui suivent:
        - build_index: Indexation
        - save_index: Vector storing
        - load_index: Charge l'indexation
        - query: Va requeter et fournir une réponse à partir d'une question utilisateur. C'est le
            coeur du RAG, utilise le store et le LLM NLP pour fournir une réponse satisfaisante.
        - is_ready: Check l'opérationnalité du système RAG très utile vu le coût de mise en place
        - document_count: Renvoi le nombre de document indexé
    """

    # =========================== LIEFECYCLE ================================
    # Décorateur qui définit une méthode qui n'a pas d'implémentation dans la classe de base
    # et qui va donc servir de modèle. ATTENTION: TOUTE CLASSE HÉRITIÈRE doit impérativement
    # réécrire ces méthodes qui, à défaut, renverront une erreur.
    # Implémenter afin de s'assurer que toutes les méthodes RAG auront exactement les mêmes
    # méthodes de base.
    @abstractmethod
    async def build_index(self, documents: list[Document]) -> None:
        """
        (Re)construit l'index vectoriel à partir d'une liste d'objets Document.\n
        C'est ici que se passe la vectorisation (Embeddings) et le storing.
        Comme c'est lourd en calcul, on ne l'appelle que lors d'une demande explicite
        de mise à jour des données.

        Parameters
        ----------
        documents (list[Document])
            Document objet prêt a être indexer (vectoriser/embedder)
        """

    @abstractmethod
    async def save_index(self) -> None:
        """
        Sauvegarde l'index actuel sur le disque pour la persistence.\n
        Evite de re-vectoriser à chaque redémarrage volontaire ou subi.
        """

    @abstractmethod
    async def load_index(self) -> bool:
        """
        Charge un index depuis le disque.\n

        Returns
        -------
        bool
            ``True`` si l'index a été trouvé sinon ``False``.
        """

    # ================== QUERY =======================================

    @abstractmethod
    async def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """
        L'utilisateur pose une question puis:
            - RETRIEVE: Le système cherche les documents.
            - AUGMENTATION: Le système envoie la question + les documents au LLM.
            - GENERATION: Le LLM répond.

        Parameters
        ----------
        question (str)
            Question de l'utilisateur.
        top_k (int)
            Nb de doc a récupérer depuis le vectore store. Défaut 5

        Returns
        -------
        dict[str, Any]
            Dictionnaire contenant à minima:
                - ``"answer"`` (``str``): Réponse NLP LLM.
                - ``"source_documents"`` (``list[Document]``): Les docs récupérés.
                - ``"scores"`` (``list[float]``): Les scores de confiance de chaque résultat.
        """

    # ========================= MONITORING =========================================

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Renvoi ``True`` si le système est opérationnel.
        """

    @abstractmethod
    def document_count(self) -> int:
        """
        Renvoi le nb de doc actuellement stocké dans l'index.
        Renvoi ``0``si l'index est vide ou non initialisé.
        """
