"""
app/core/config.py

Centralise les variables d'environnement. Au lieu que chaque fichier appel ses propres variables
d'environnement, on utilise ce fichier qui va lire les .env et fallback si ceux-ci n'ont pas la
la variable ou est mal entré avec une valeur par défaut.\n
On se protège avec Pydantic.

Tous les modules importent get_settings() au lieu de faire os.environ.

Le fichier .env n'est qu'un fichier texte brut.\n
Utiliser une classe Settings (via pydantic-settings) apporte trois avantages majeurs
en Data Science et Production :

1. Validation de Type
    Si openagenda_max_events=BEAUCOUP dans .env, Pydantic lèvera
    une erreur immédiatement au démarrage car il attend un int.

2. Auto-complétion (IDE)
    Dans le code, quand on écrit settings.,
    l'éditeur va proposer mistral_api_key != avec un simple os.getenv, on navigues à
    l'aveugle donc risque d'erreur.

3. Valeurs par défaut
    On peut mettre des valeurs fallback (ex: app_port: int = 8000).
    Si la variable est absente du .env, l'app ne plante pas.

4. Conversion automatique
    Il transforme la chaîne "99.99" du .env en un véritable float
    utilisable pour tes calculs géographiques.
"""

# imports
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

from pydantic_settings import BaseSettings, SettingsConfigDict

# =====================================================================


class Settings(BaseSettings):
    """
    Classe paramétrique qui centralise les variables d'environnement. Lit automatiquement le .env
    dans le dossier courant si l'argument env_file est déclaré dans model_config via
    Pydantic-settings ce qui permet en plus un contrôle.
    """

    # ======================== LLM =====================================
    mistral_base_url: str = "https://api.mistral.ai/v1"  # Nécéssaire pour ragas
    mistral_api_key: str
    """OBLIGATOIRE: CLEF API POUR MISTRAL (CRASH SI ABSENT)"""
    llm_model: str = "devstral-small-latest"
    """'devstral-small-latest', 'devstral-medium-latest' ou 'devstral-latest'"""
    llm_temperature: float = 0.1
    """Contrôle le côté factuel (faible valeur, conseillé 0.2) ou imaginaire du modèle """
    llm_max_tokens: int = 1024
    """Nb max de tokens à générer """
    llm_top_p: float = 0.9
    """Contrôle le ciblage des résultats (valeur conseillé 0.9)"""
    embed_model: str = "mistral-embed"
    chunk_size: int = 500
    chunk_overlap: int = 100
    # =================== OpenAgenda ===================================
    openagenda_public_url: str
    """OBLIGATOIRE: URL OPEN AGENDA POUR FETCH DONNEES (CRASH SI ABSENT)"""
    openagenda_updatedat: Optional[Union[List[str], str]] = "2025"
    openagenda_location_city: Optional[Union[List[str], str]] = "Paris"
    openagenda_location_region: Optional[Union[List[str], str]] = "Île-de-France"
    openagenda_limit: int = 20
    """Limite du nombre de résultats"""
    openagenda_offset: int = 0
    """Index du premier résultat renvoyé"""
    openagenda_lang: Optional[str] = "fr"
    """Code langue de 2 lettres"""
    openagenda_timezone: Optional[Union[List[str], str]] = "Europe/Berlin"
    """Le fuseau horaire utilisé pour interpréter les dates et heures dans la requête et
    les données de la réponse."""
    openagenda_max_events: int = 50
    """Nb max events a ingérer par ré-indexation (/rebuild)."""

    # ================== RAG ==========================================
    rag_top_k: int = 5
    """Nb de doc par requête FAISS"""
    faiss_index_path: Path = Path("data/faiss_index")
    """Chemin du dossier persistant pour l'indexation FAISS."""

    # ================ Securité /rebuild =======================================
    rebuild_api_key: str
    """OBLIGATOIRE: CLEF API POUR POUVOIR INDEXER"""

    # ================ Serveur ===============================================
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"
    """'developpement' ou 'production'"""

    # ================= CONFIG ===============================================
    model_config = SettingsConfigDict(
        env_file=(".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore les variables inconnues
        coerce_numbers_to_str=True,  # Permet la conversion des int en str auto
    )


# ====================================================================


# C'est un cache qui garde en mémoire le résultat du premier appel.
# La première fois qu'on appelle 'get_settings()', Python lit le fichier '.env' et
# crée l'objet 'Settings'. Toutes les fois suivantes, il retourne **exactement le même objet**
# sans relire le fichier. C'est le pattern **Singleton** — un seul objet de configuration pour
# toute l'application.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retourne le singleton 'Settings' caché.\n
    L'utilisation de lru_cache garanti que l'.env n'est lu qu'une fois et que tous les modules
    recoivent exactement le même objet.
    """
    return Settings()  # type:ignore # pragma: no cover
