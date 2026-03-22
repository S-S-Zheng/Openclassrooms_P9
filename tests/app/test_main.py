"""
tests/app/test_main.py

Suite de tests pour le point d'entrée principal (main.py) et le cycle de vie de l'application.

Ce module valide les fonctionnalités de base de l'infrastructure FastAPI :
1. La disponibilité opérationnelle via le endpoint de santé (Healthcheck).
2. La redirection de l'URL racine vers l'interface de documentation Swagger.
3. Le bon fonctionnement du 'lifespan', garantissant que le RAG et les paramètres sont
    correctement chargés en mémoire au démarrage du serveur.
"""

# Imports
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

# =================== Health =======================


@pytest.mark.unit
# On s'assure que /health est fonctionnelle: code 200
# On s'assure que la réponse est bien status:ok
def test_healthcheck(client):
    """
    Vérifie que le point d'entrée de santé est opérationnel.

    Indispensable pour les sondes de disponibilité (Liveness/Readiness probes)
    dans les environnements de déploiement type Docker ou Kubernetes.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# =================== Root =======================


@pytest.mark.unit
def test_root_redirects_to_docs(client):
    """
    Vérifie la redirection automatique de la racine.

    S'assure que tout utilisateur accédant à l'URL de base est immédiatement
    orienté vers la documentation interactive de l'API (Swagger UI).
    """
    # follow_redirects=False permet de vérifier le code 307 de redirection
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


# =================== Lifespan =======================


# ---------------- Test du Lifespan réussi
@pytest.mark.integration
def test_lifespan_startup_success(fake_settings):
    """
    Valide l'initialisation du contexte de l'application.

    Vérifie que le mécanisme 'lifespan' a correctement injecté ce qu'il faut dans les ``app.state``
    """
    app = create_app()

    # On mocke EventRAGPipeline pour éviter de charger les vrais modèles
    with patch("app.main.EventRAGPipeline") as MockRAG:
        # On simule un chargement d'index réussi
        MockRAG.return_value.load_index = AsyncMock(return_value=True)
        # TestClient déclenche le lifespan à l'entrée du bloc 'with'
        with TestClient(app) as client:
            assert client.app.state.index_ready is True  # type:ignore
            assert client.app.state.rag is not None  # type:ignore
            MockRAG.return_value.load_index.assert_called_once()


# ---------------- Test du Lifespan (Mode dégradé)
@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_success_degraded(mock_client_unready):
    """
    Vérifie que le serveur répond au healthcheck même si l'index est absent.
    Utilise la fixture 'mock_client_unready' définie dans conftest.
    """
    response = mock_client_unready.get("/health")

    # Assertions
    assert mock_client_unready.app.state.index_ready is False
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------- Test que le shutdown s'execute bien
@pytest.mark.integration
@pytest.mark.asyncio
async def test_lifespan_shutdown(caplog):
    """Vérifie que le log de nettoyage est présent à la fermeture."""
    app = create_app()

    with TestClient(app):
        pass  # Startup... puis Shutdown automatique à la sortie du bloc

    # Assertions
    assert "Arrêt du serveur, nettoyage..." in caplog.text
