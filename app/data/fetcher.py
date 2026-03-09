"""
app/data/fetcher.py

Client HTTP async pour l'API publique Opendatasoft d'OpenAgenda (pas besoin de clef API, juste URL)

On ne fait que fetcher la donnée brute des events telles qu'ils apparaissent dans l'output
de l'onglet API de https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/
evenements-publics-openagenda/records. Le traitement se fait dans processor.py.
"""

# imports
import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import Settings

logger = logging.getLogger(__name__)


# ================================================================


class OpenAgendaFetcher:
    """
    Client async pour l'API publique OpenAgenda.\n
    Utilise httpx.AsyncClient en background pour ne pas bloquer la boucle des events pendant I/O.\n
    La pagination est gérée le param offset.

    Parameters
    ----------
    settings:
        Les param d'application (URL + filtres).
    """

    def __init__(self, settings: Settings) -> None:
        # self._key = settings.openagenda_public_key
        self._url = settings.openagenda_public_url
        self._max_events = settings.openagenda_max_events
        self._year = settings.openagenda_updatedat
        self._city = settings.openagenda_location_city
        self._region = settings.openagenda_location_region
        self._limit = settings.openagenda_limit
        self._offset = settings.openagenda_offset
        self._lang = settings.openagenda_lang
        self._timezone = settings.openagenda_timezone

    # Méthode de fetching de la data brute
    async def fetch_events(self) -> list[dict[str, Any]]:
        """
        Récupère tous les events jusqu'à atteindre la limite _max_events ou que
        l'ensemble ait été récupéré.

        Returns
        -------
        list[dict[str, Any]]
            Le même dictionnaire brute visible dans l'output d'OpenAgenda.
        """
        collected: list[dict[str, Any]] = []
        current_offset = self._offset

        async with httpx.AsyncClient(timeout=30.0) as client:
            while len(collected) < self._max_events:
                page = await self._fetch_page(client, current_offset)
                results: list[dict[str, Any]] = page.get("results", [])

                if not results:
                    logger.info("Plus d'event OpenAgenda, fin de la pagination.")
                    break

                collected.extend(results)
                logger.info(f"Récupéré {len(collected)} événements")

                current_offset += self._limit
                # Sécurité pour ne pas dépasser le nombre total disponible
                if len(collected) >= page.get("total_count", 0):
                    break

        # Nettoye suivant le max d'events
        return collected[: self._max_events]

    # ====================== HELPER =============================
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )  # Décorateur pour ré essayer en augmentant les temps d'attente
    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        offset: int,  # On accepte l'offset dynamique ici
    ) -> dict[str, Any]:
        """
        Récupère une page unique d'events depuis l'API OpenAgenda.
        Le décorateur issu de tenacity permet 3 essais avec un temps d'attente exponentiel après
        chaque tentative.

        Parameters
        ----------
        client:
            httpx.AsyncClient.

        Returns
        -------
        dict[str, Any]
            Corpus brute de la réponse API
        """
        where_clause = (
            f'location_city:"{self._city}" AND location_region:"{self._region}" AND "{self._year}"'
        )
        params: dict[str, Any] = {
            "order_by": "updatedat desc",  # Trier par les plus récents
            # Le paramètre 'where' est plus puissant que 'refine' pour les filtres complexes
            "where": where_clause,
            # "updatedat": self._year,
            # "location_city": self._city,
            # "location_region": self._region,
            # # On peut toujours utiliser 'refine' pour les facettes simples
            # "refine": [
            #     f"location_region:{self._region}"
            # ],
            "limit": self._limit,
            "offset": offset,  # Utilise l'offset passé par la boucle
            "lang": self._lang.lower() if self._lang is not None else self._lang,
            "timezone": self._timezone,
        }
        response = await client.get(self._url, params=params)

        if response.status_code != 200:
            logger.error(f"Erreur API: {response.status_code}")
            response.raise_for_status()

        return response.json()
