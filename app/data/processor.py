"""
app/data/processor.py

Transforme la donnée brute de fetcher.py (dico) en objets "Document" LangChain
prêts pour l'indexation FAISS.

Ce fichier accompli les tâches suivantes:
    - Normalisation du texte
    - Extraction des métadonnées et découpage (chunking) optionnel
Pas d'appels HTTP, pas d'appels LLM — transformation de données pure.
"""

# imports
import logging
import re
from pathlib import Path
from typing import Any

import yaml

# C'est l'unité de base de LangChain.
# Un document contient du texte (page_content) et des infos annexes (metadata)
from langchain_core.documents import Document

# Découpe intelligemment le texte (en essayant de ne pas couper au milieu d'une phrase)
# pour que le LLM puisse le lire sans être submergé.
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings

logger = logging.getLogger(__name__)

MAP_FILE = Path(__file__).parent / "mapping.yaml"

# ===========================================================


class EventDocumentProcessor:
    """
    Convertit une liste d'événements bruts en objets "Document" en découpant les descriptions
    trop longues.

    Parameters
    ----------
    chunk_size:
        Nb de caracatères max par chunk
    chunk_overlap:
        Nb de caractère de chevauchement (caractères partagé par deux chunks consecutifs)
    """

    def __init__(self, settings: Settings) -> None:
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "; ", ", ", ": ", ""],
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )  # ordre de priorité de la découpe suivant les séparateurs (double saut, simple, ., ...)
        self.mapping = self._load_map()  # Charge le yaml de mapping

    # Méthode pour convertir brut vers langchain
    def process(self, raw_events: list[dict[str, Any]]) -> list[Document]:
        """
        -----------------
        # Penser a implementer:\n
        # Utilise tous les coeurs du CPU pour transformer les documents en parallèle
            with ProcessPoolExecutor() as executor:
        ----------------
        Transforme les données brutes en liste de Documents LangChain.\n
        Chaque event est converti en un bloc de texte structuré puis chunké si le texte dépasse
        la limite fixée par chunk_size.

        Parameters
        ----------
        raw_events:
            Liste des events brutes renvoyés par OpenAgendaFetcher

        Returns
        -------
        list[Document]
            Liste aplatie d'objets "Document"
        """
        documents: list[Document] = []

        for event in raw_events:
            try:
                doc = self._event_to_document(event)
                # Si l'événement est trop pauvre en info, doc peut être None
                if not doc:
                    continue
                # Découpage si le contenu dépasse la limite
                if len(doc.page_content) > self._chunk_size:
                    # Split_documents conserve les métadonnées sur chaque chunk
                    chunks = self._splitter.split_documents([doc])
                    documents.extend(chunks)
                else:
                    documents.append(doc)
            except Exception as exc:
                # Malformed events are skipped, not fatal
                logger.warning(f"Événement ignoré (uid={event.get('uid')}): {exc}")
                continue

        logger.info(f"Traitement terminé : {len(raw_events)} bruts --> {len(documents)} docs.")
        return documents

    # ============================== HELPER ===============================

    def _load_map(self) -> dict:
        """
        Charge le yaml de mapping de la donnée
        """
        with open(MAP_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _event_to_document(self, event: dict[str, Any]) -> Document:
        """
        Transforme le dictionnaire brute de l'API en un objet Document propre.
        Il isole la logique de mapping. Si le nom d'un champ change dans l'API,
        vous ne changez que cette fonction.

        Parameters
        ----------
        event:
            Dictionnaire d'un unique event brute

        Returns
        -------
        Document
            Document langchain avec page_content et metadata
        """
        # --- Extraction des données (API publique) ---
        # Informations principales
        # -------------------------
        # title: str = event.get("title_fr") or ""
        # description: str = event.get("description_fr") or ""
        # details: str = \
        #     event.get("longdescription_fr") or ""
        # # Localisation
        # address: str = event.get("location_address") or ""
        # city: str = event.get("location_city") or ""
        # venue: str = event.get("location_name") or ""
        # # Couverture temporelle
        # beginning: str = event.get("firstdate_begin") or ""
        # ending: str = event.get("lastdate_end") or ""
        # # Accessibilité
        # handicap : list[dict[str, Any]] = event.get("accessibility_label_fr") or []
        # age_min: int = event.get("age_min") or 0
        # age_max: int = event.get("age_max") or 200
        # # Contact
        # phone: str = event.get("location_phone") or ""
        # website: str = event.get("location_website") or ""
        # ----------------------------------------------------
        # Bulk extraction depuis le YAML
        raw_data = {
            field["label"]: event.get(field["api_key"], field["default"])
            for field in self.mapping["event_fields"]
        }

        # --- Construction du bloc de texte pour le LLM ---
        # ----------------------------------------------------
        # # Mistral a été entrainé en NLP donc il est plus performant en présence de texte
        # # structuré pour l'homme que des json et autres
        # parts = [f"Titre : {title}"]
        # if beginning or ending: parts.append(f"Dates : {beginning} au {ending}")
        # if venue or city: parts.append(f"Lieu : {venue}, {address}, {city}")
        # if description: parts.append(f"Description : {description}")
        # if details and len(details) > len(description):
        #     # Nettoyage sommaire des balises HTML si présentes
        #     clean_details = re.sub('<[^<]+?>', '', details)
        #     parts.append(f"Détails : {clean_details}")
        # if handicap : parts.append(f"Accessibilités : {handicap}")
        # if age_min or age_max: parts.append(f"Tranche d'âge : {age_min} à {age_max}")
        # if phone: parts.append(f"Téléphone : {phone}")
        # if website: parts.append(f"Site web : {website}")
        # ----------------------------------------------------
        # On sécurise titre et description
        title = str(raw_data.get("Titre", "Sans titre")).strip()
        description = str(raw_data.get("Description", "")).strip()
        keys_to_skip = ["Titre", "Description"]
        # On élimine les clés inutiles d'un coup
        keys_to_delete = [
            key
            for key, val in raw_data.items()
            if val in (None, "", [])
            or (key == "Age_min" and val < 0)
            or (key == "Age_max" and val >= 200)
            or (key in keys_to_skip)
        ]
        for key in keys_to_delete:
            del raw_data[key]
        # Nettoyage et formatage final de chaque champ
        final_parts = [
            # Suppr les balise HTML si str + <
            f"{key} : {re.sub(r'<[^<]+?>', '', str(val))}"
            if isinstance(val, str) and "<" in val
            # Convert les listes (ex: handicap) en str séparées par ,
            else f"{key} : {', '.join(map(str, val))}"
            if isinstance(val, list)
            # Le reste (int, float, str hors HTML) -> Affichage tel quel
            else f"{key} : {val}"
            for key, val in raw_data.items()
        ]

        # Le contenu qui sera vectorisé au final
        # page_content = "\n".join(final_parts)
        # On place le titre et la description en premier pour qu'ils soient dans le premier chunk
        content_header = f"Titre : {title}\nDescription : {re.sub(r'<[^<]+?>', '', description)}\n"
        content_body = "\n".join(final_parts)

        page_content = f"{content_header}\n\n--- Détails ---\n{content_body}"

        # --- Métadonnées pour la source (pas vectorisé) ---
        # ----------------------------------------------------
        # metadata: dict[str, Any] = {
        #     "uid": event.get("uid"),
        #     "url": event.get("canonicalurl"),
        #     "title": title,
        # }
        # ----------------------------------------------------
        # ----------------------------------------------------
        # metadata: dict[str, Any] = {
        #     "uid": event.get(self.mapping["metadata_fields"]["uid"]),
        #     "url": event.get(self.mapping["metadata_fields"]["url"]),
        #     "title": raw_data.get("Titre", "Sans titre"),
        # }
        # ----------------------------------------------------
        metadata = {
            key: event.get(api_key)
            for key, api_key in self.mapping["metadata_fields"].items()
            if event.get(api_key) is not None
        }

        return Document(page_content=page_content, metadata=metadata)
