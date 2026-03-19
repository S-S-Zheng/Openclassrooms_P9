"""
tests/test_fetcher.py

tests concernant la pipeline ETL partie Fetching ``OpenAgendaFetcher``

"""

# imports
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.etl.fetcher import OpenAgendaFetcher

# ===========================================================


# On marque toute la classe comme unitaire car on mocke tout l'externe (HTTP)
@pytest.mark.unit
class TestOpenAgendaFetcher:
    # --------------------------------- Instance Fetcher avec fake_settings
    @pytest.fixture
    def fetcher(self, fake_settings) -> OpenAgendaFetcher:
        """Fetcher configuré avec les fake_settings du conftest."""
        return OpenAgendaFetcher(fake_settings)

    # --------------------------------- Vérifie qu'une page simple est correctement récupérée
    @pytest.mark.asyncio
    async def test_fetch_single_page(self, fetcher):
        """
        Quand l'API renvoi une page unique avec 2 events et le total_count est bien atteint
        ``fetch_events`` doit renvoyer exactement ces deux seuls events.
        """
        fake_response_body = {
            "results": [
                {"uid": 1, "title_fr": "Événement A"},
                {"uid": 2, "title_fr": "Événement B"},
            ],
            "total_count": 2,
        }

        # On patche httpx dans le module où il est utilisé
        with patch("app.etl.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            mock_resp = MagicMock(status_code=200)
            mock_resp.json.return_value = fake_response_body
            mock_client.get = AsyncMock(return_value=mock_resp)

            events = await fetcher.fetch_events()

        # Assertions
        assert len(events) == 2
        assert events[0]["uid"] == 1
        assert events[1]["title_fr"] == "Événement B"

    # --------------------------------- Vérifie que max_events est bien respectée
    @pytest.mark.asyncio
    async def test_fetch_respects_max_events(self, fake_settings):
        """
        Vérifie que la limite globale _max_events est respectée.
        """
        fake_settings.openagenda_max_events = 2
        fetcher = OpenAgendaFetcher(fake_settings)

        # L'API en renvoie 5
        fake_data = {
            "results": [{"uid": i} for i in range(5)],
            "total_count": 5,
        }

        with patch("app.etl.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_resp = MagicMock(status_code=200)
            mock_resp.json.return_value = fake_data
            mock_client.get = AsyncMock(return_value=mock_resp)

            events = await fetcher.fetch_events()

        # Assertions
        # On a eu 5 mais le fetcher doit tronquer a 2
        assert len(events) == 2

    # --------------------------------- Vérifie que le fetcher cherche bien a atteindre total_count
    @pytest.mark.asyncio
    async def test_fetch_paginates(self, fake_settings):
        """
        Vérifie que le fetcher boucle tant que total_count n'est pas atteint.
        """
        fake_settings.openagenda_max_events = 10
        fake_settings.openagenda_limit = 2
        fetcher = OpenAgendaFetcher(fake_settings)

        page1 = {"results": [{"uid": i} for i in range(4)], "total_count": 9}
        page2 = {"results": [{"uid": i + 4} for i in range(5)], "total_count": 9}

        with patch("app.etl.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            mock_resp1 = MagicMock(status_code=200)
            mock_resp1.json.return_value = page1
            mock_resp2 = MagicMock(status_code=200)
            mock_resp2.json.return_value = page2

            mock_client.get = AsyncMock(side_effect=[mock_resp1, mock_resp2])

            events = await fetcher.fetch_events()

        # Assertions
        assert len(events) == 9
        # Vérifie que le client a été appelé deux fois (pagination active)
        assert mock_client.get.call_count == 2

    # --------------------------------- Vérifie que la classe est stable si la réponse API est vide
    @pytest.mark.asyncio
    async def test_fetch_empty_response(self, fetcher):
        """ "Vérifie la sortie propre si l'API est vide."""
        with patch("app.etl.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_resp = MagicMock(status_code=200)
            mock_resp.json.return_value = {"results": [], "total_count": 0}
            mock_client.get = AsyncMock(return_value=mock_resp)

            events = await fetcher.fetch_events()

        assert events == []

    # --------------------------------- Vérifie la gestion des erreurs API (404, 500, etc.)
    @pytest.mark.asyncio
    async def test_fetch_events_raises_for_status(self, fetcher):
        """Vérifie que fetch_events lève une exception et loggue en cas d'erreur 500."""

        with patch("app.etl.fetcher.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            # On crée une réponse simulant une Erreur Interne du Serveur (500)
            mock_resp = MagicMock(status_code=500)
            # On simule le comportement de httpx : raise_for_status() lève une erreur si != 200
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Erreur Interne (500)", request=MagicMock(), response=mock_resp
            )
            mock_client.get = AsyncMock(return_value=mock_resp)

            # On vérifie que l'appel lève bien l'exception attendue
            with pytest.raises(httpx.HTTPStatusError):
                await fetcher.fetch_events()

            # Assertions
            # Vérifier que raise_for_status a bien été appelé le nb de fois que retry demande
            assert mock_resp.raise_for_status.call_count == 3
