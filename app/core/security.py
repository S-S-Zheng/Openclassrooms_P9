"""
app/core/security.py

Dépendance FastAPI qui sécurise l'endpoint /rebuild avec une clef API partagée (shared-secret)
transmise via le header HTTP X-Rebuild-Key.

Remarques:
    - Utilise secrets.compare_digest pour une comparaison continuelle afin de se prémunir des
        attaques par canal auxiliaire (timing attacks)
    - Renvoie une erreur 403 (pas 401) car pas de défi www-authenticate; le client connait
        deja le format du secret

Dans un projet RAG, la reconstruction de l'index FAISS est une opération très couteuse (appel API
Mistral, calculs d'embeddings, CPU) et sensible. Si l'endpoint n'est pas locké et que tout le monde
pouvait l'utiliser, les attaques Dos seraient terrible.
"""

# Imports
# Concu pour la cryptographie, au lieu de faire ==, la fonction secrets.compare_digest(a,b) comapre
# deux chaines en prenant toujours le meme temps, peu importe ou se trouve l'erreur ce qui protege
# d'une attaque temporelle.
import secrets

# L'utilisation de 'Security' au lieu de 'Depends' permet d'intégrer
# automatiquement le bouton 'Authorize' dans la doc Swagger.
# from fastapi import Header
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

# ========================================================================

# On définit le schéma de sécurité pour Swagger
# name="X-Rebuild-Key" est le nom exact de l'en-tête attendu
rebuild_key_scheme = APIKeyHeader(
    name="X-Rebuild-Key",
    auto_error=False,
    description="Clef API nécéssaire pour trigger l'indexation FAISS (/rebuild)",
)


async def require_rebuild_key(
    # x_rebuild_key: str = Header(
    #     ...,
    #     alias="X-Rebuild-Key",
    #     description="Clef API nécéssaire pour trigger l'indexation FAISS (/rebuild)",
    # ),  # Header indique a FastApi de chercher la val dans header HTTP des requetes et pas l'URL
    x_rebuild_key: str = Security(rebuild_key_scheme),
) -> str:
    """
    Dépendance FastAPI injectée dans la route /rebuild.

    Parameters
    ----------
        x_rebuild_key(str): Valeur du header X-Rebuild-Key de la requête.

    Exceptions
    ----------
        HTTPException 403 si la clé fournie ne correspond pas à REBUILD_API_KEY.
    """
    expected = get_settings().rebuild_api_key
    # if not secrets.compare_digest(x_rebuild_key, expected):
    #     raise HTTPException(
    #         status_code=status.HTTP_403_FORBIDDEN,
    #         detail="Clé API /rebuild invalide ou manquante.",
    #     )  # HTTP_403_FORBIDDEN == 'Je sais qui tu es/veux mais tu n'as pas le droit d'être ici'

    # On vérifie si la clé est présente et correcte
    if not x_rebuild_key or not secrets.compare_digest(x_rebuild_key, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé API /rebuild invalide ou manquante.",
        )

    return x_rebuild_key
