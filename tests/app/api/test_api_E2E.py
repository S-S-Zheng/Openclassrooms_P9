"""
tests/app/api/test_api_E2E.py

Test fonctionnel E2E de l'API afin de vérifier que tous les composants ont été bien intégrés
et fonctionnent correctement, on va pour cela utiliser le flux le plus long:
1. Vérifier que /ask renvoie 503 au démarrage (index vide).
2. Déclencher un /rebuild avec succès.
3. Vérifier que /ask renvoie 200 avec des sources après le rebuild.

Note:
-----
Un test E2E devrait normalement utiliser les vrais composants afin de vérifier que la logique
métier/technique soit en plus bien respectée mais dans le cas du RAG, on évite
car cela impliquerait sinon de consommer des crédits LLM, utiliser les vrais clefs de protections
et réaliser des embeddings qui peuvent être très long (Métier vérifiée par eval_rag).\n
Test réalisé pour répondre à la demande du projet d'avoir un test fonctionnel api_test.py
ce test est sinon redondant car reprend quasi exactement les tests ask et rebuild...
"""

# Imports
from unittest.mock import AsyncMock, patch

import pytest

# ===================================================


@pytest.mark.functional
@pytest.mark.asyncio
@patch("app.api.routes.rebuild.OpenAgendaFetcher")
@patch("app.api.routes.rebuild.EventDocumentProcessor")
async def api_E2E(
    mock_fetcher_class,
    mock_processor_class,
    mock_client_unready,
    fake_settings,
    raw_events,
    sample_documents,
):
    # ----------- ÉTAPE 1 - On démarre avec un état d'index non chargé
    client = mock_client_unready
    # rejet de /ask avec code 503
    response_init = client.post("/ask", json={"question": "C'est quoi le surf ?"})
    assert response_init.status_code == 503
    assert "Pas d'index trouvé, démarrage en mode dégradé" in response_init.json()["detail"]

    # ------------ ÉTAPE 2 - /rebuild
    # ******** Phase ETL
    # Le Fetcher renvoie nos raw_events
    mock_fetcher = mock_fetcher_class.return_value
    mock_fetcher.fetch_events = AsyncMock(
        return_value=raw_events
    )  # ``fetch_event()`` -> asynchrone
    # Le Processor renvoie nos sample_documents
    mock_processor = mock_processor_class.return_value
    mock_processor.process.return_value = sample_documents  # ``process()`` -> synchrone
    # ******** Phase index
    mock_rag = client.app.state.rag
    mock_rag.build_index = AsyncMock()
    mock_rag.save_index = AsyncMock()

    # Appel de la route
    client.app.state.settings = fake_settings
    headers = {"X-Rebuild-Key": fake_settings.rebuild_api_key}
    response = client.post("/rebuild", headers=headers)

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["documents_indexed"] == len(sample_documents)
    assert client.app.state.index_ready is True

    # ------------ ÉTAPE 3 - /ask fonctionne
    # On configure la réponse que le RAG doit donner à la question
    mock_rag.query = AsyncMock(
        return_value={
            "answer": "Peyo Lizarazu parle de surf.",
            "source_documents": [sample_documents[0]],
            "scores": [0.98764321],
        }
    )
    # Exécution de la requête
    payload = {"question": "Qui est Peyo ?", "top_k": 1}
    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "Peyo Lizarazu" in data["answer"]
    assert data["sources"][0]["event_title"] == "Dédicace de Peyo Lizarazu"
