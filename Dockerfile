# --- Étape 1 : Build (Installation des dépendances) ---
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true

# Installation des dépendances système nécessaires à la compilation (gcc, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation de Poetry 2.0
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

# Copie des fichiers de config uniquement (pour le cache Docker)
COPY pyproject.toml poetry.lock ./ 
# Note : on ne copie pas encore le code pour optimiser le cache des layers
RUN /root/.local/bin/poetry install --only main --no-root

# --- Étape 2 : Runtime (Image finale légère) ---
FROM python:3.12-slim AS runtime

# Installation des dépendances système pour l'exécution
# libshared-intel-lp64 est souvent requis par FAISS sur Debian/Ubuntu
# libgomp1 est crucial pour le parallélisme des calculs d'embeddings
# libopenblas-dev car FAISS repose sur des calculs matriciels intensifs.
# Sans cette librairie (ou une équivalente) -> erreurs de segmentation ou des performances
# CPU catastrophiques.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Création de l'utilisateur non-root
RUN useradd -m -u 1000 raguser
USER raguser
ENV PATH="/app/.venv/bin:/home/raguser/.local/bin:${PATH}" \
    PYTHONPATH="/app"

WORKDIR /app

# On récupère l'environnement virtuel créé à l'étape précédente
COPY --from=builder --chown=raguser /app/.venv /app/.venv
# On copie le code source
COPY --chown=raguser . .

EXPOSE 8000

# ---------------- EN LOCAL ----------------
# # Commande de lancement
# CMD python -m app.db.create_db && \
#     python -m app.db.import_dataset_to_db && \
#     uvicorn app.main:app --host 0.0.0.0 --port 7860
# On ne garde que le lancement de l'API.
# ou
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# ---------------- AVEC SUPABASE ---------------------
# L'initialisation de la DB se fait une seule fois manuellement ou via une migration.
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]