# Projet 9: Concevez et déployez un système RAG

> Système de **Retrieval-Augmented Generation** (RAG) exposé via une API REST,
> répondant aux questions en langage naturel sur les événements culturels
> en Île-de-France à partir des données **OpenAgenda**.

<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- A décommenter si besoin dans le futur
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/S-S-Zheng/Openclassrooms_P9.git">
    <!-- <img src="images/logo.png" alt="Logo" width="80" height="80"> -->
  </a>

<h3 align="center">Projet 9: Concevez et déployez un système RAG</h3>

  <p align="center">
    Projet 9 de la formation d'OpenClassrooms: Data scientist Machine Learning (projet débuté le 05/03/2026)
    <br />
    <a href="https://github.com/S-S-Zheng/Openclassrooms_P9.git"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/S-S-Zheng/Openclassrooms_P9.git">View Demo</a>
    &middot;
    <a href="https://github.com/S-S-Zheng/Openclassrooms_P9.git/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/S-S-Zheng/Openclassrooms_P9.git/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Sommaire</summary>
  <ol>
    <li><a href="#a-propos-du-projet">A propos du projet</a></li>
      <ul>
        <li><a href="#objectifs">Objectifs</a></li>
        <li><a href="#architecture-du-projet">Architecture du projet</a></li>
        <li><a href="#construit-avec">Construit avec</a></li>
      </ul>
    <li>
      <a href="#getting-started">Getting started</a>
      <ul>
        <li><a href="#prerequis">Prerequis</a></li>
        <li><a href="#installation-et-configuration">Installation et configuration</a></li>
        <ul>
          <li><a href="#cloner-le-dépôt">Cloner le dépôt</a></li>
          <li><a href="#installer-lenvironnement">Installer l'environnement</a></li>
          <li><a href="#test-local">Test local</a></li>
        </ul>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
      <ul>
        <li><a href="#exemple-de-requête-via-curl">Exemple de requête via cURL</a></li>
        <li><a href="#documentation-interactive">Documentation Interactive</a></li>
      </ul>
    <!-- <li><a href="#deployment">Deploiement</a></li>
    <li><a href="#authentication">Authentification</a></li> -->
    <li><a href="#démarrage-avec-docker">Démarrage avec Docker</a></li>
    <!-- <li><a href="#security">Securité</a></li> -->
    <li><a href="#évaluation-ragas">Évaluation (RAGAS)</a></li>
      <ul>
        <li><a href="#qa">QA</a></li>
        <li><a href="#métriques-utilisées">Métriques utilisées</a></li>
        <li><a href="#lancer-lévaluation">Lancer l'évaluation</a></li>
      </ul>
    <li><a href="#documentation_technique">Documentation technique</a></li>
      <ul>
        <li><a href="#architecture-de-système">Architecture du système</a></li>
        <li><a href="#préparation-et-vectorisation">Préparation et vectorisation</a></li>
        <li><a href="#choix-du-modèle-nlp">Choix du modèle NLP</a></li>
        <li><a href="#base-vectorielle-faiss">Base vectorielle (FAISS)</a></li>
        <li><a href="#exemple-de-résultats">Exemple de résultats</a></li>
      </ul>
    <li><a href="#pistes-damélioration">Pistes d'amélioration</a></li>
      <ul>
        <li><a href="#qualité-rag">Qualité RAG</a></li>
        <li><a href="#infrastructure">Infrastructure</a></li>
      </ul>
    <!-- <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Participations</a></li> -->
    <li><a href="#license">License</a></li>
    <!-- <li><a href="#acknowledgments">Remerciements</a></li> -->
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## A propos du projet

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

"Vous êtes data scientist freelance, spécialisé dans le traitement du langage naturel et la création de systèmes intelligents. Vous intervenez en mission pour Puls-Events, une entreprise technologique qui développe une plateforme de recommandations culturelles personnalisées.

Puls-Events souhaite tester un nouveau chatbot intelligent capable de répondre à des questions utilisateurs sur les événements culturels à venir, en s'appuyant sur un système RAG (Retrieval-Augmented Generation) combinant recherche vectorielle et génération de réponse en langage naturel.

Votre mission est de livrer un POC (Proof of Concept) complet, avec une API exploitable par les équipes produit et marketing. Ce POC devra démontrer la faisabilité technique, la pertinence métier et la performance du système."

Le projet incluera:

1. Un dépôt GitHub contenant le code, l'environnement et la documentation.
2. Une API et sa documentation pour exposer le modèle.
3. Un ensemble de tests unitaires et fonctionnels servant à garantir la fiabilité et la robustesse.

### Objectifs du projet

* **Problématique** : Comment permettre à un utilisateur de trouver un événement dans une base de données mouvante (OpenAgenda) sans recherche par mots-clés rigide ?
* **Valeur métier** : Optimisation du temps de recherche et de sa fructivité en étant orienté par un chatbot.
* **Périmètre** : Événements culturels (concert, théâtre, exposition, salon, manifestation, colloque, compétition, rencontre, art, sport) en **Île-de-France, Paris**, année **2026**.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Architecture du projet

```text
livrable_P9
├── app/                                      # Cœur de l'application (FastAPI)
│   ├── api/                                  # Couche interface
│   │   ├── routes/                           # Points d'entrée (Endpoints) de l'API
│   │   │   ├── ask.py                        # /ask/
│   │   │   └── rebuild.py                    # /rebuild/
│   │   ├── dependencies.py                   # Garde fou si RAG non opérationnel
│   │   └── schemas.py                        # Validation des données (Pydantic)
│   ├── core/                                 # Paramètres et sécurité de l'app
│   │   ├── config.py                         # Paramètres d'environnement avec lru_cache
│   │   └── security.py                       # Vérouillage /rebuild
│   ├── etl/                                  # Extraction-Transformation-Loading
│   │   ├── fetcher.py                        # Classe pour Récupération    
│   │   ├── mapping.yaml                      # Mapping du contextuelle et metadatas
│   │   └── processor.py                      # Classe Pré-traitement
│   ├── rag/                                  # RAG
│   │   ├── base_rag.py                       # Base class pour systèmes RAG   
│   │   ├── prompt_management.txt             # Texte pour donner contexte et consignes au LLM
│   │   └── rag_pipeline.py                   # Classe RAG (index, save/load index et query)
│   ├── utils/                                # Utilitaires transverses
│   │   └── save_load_datas.py                # Script de sauvegarde/chargement de données
│   └── main.py                               # Point d'entrée principal de l'API
├── CLI                                       # Command Line Interface
│   ├── build_index.py                        # ETL + Indexation
│   ├── eval_rag.py                           # A partir d'un json QA, évalue le RAG
│   └── qa_helper.py                          # Extrait des documents indexés pour aider au QA
├── datas/                                    # datasets, résultats, modèles et autres fichiers
│   └── raw/
│       └── qa_pairs.json                     # Pairs de questions/ground_truth pour éval
├── tests/                                    # Suite de tests automatisée (Pytest)
├── coverage.ini                              # Configuration du rapport de couverture
├── Dockerfile                                # Instructions de conteneurisation
├── docker-compose.yml                        # Constructeur du docker
├── .gitignore                                # Elements à ignorer pour le dépôt GitHub
├── .dockerignore                             # Elements à ignorer pour le/les conteneurs
├── .env.example                              # Exemple mini du .env(ajouter .env.test aussi!)
├── LICENSE                                   # Licence MIT du projet
├── pytest.ini                                # Configuration globale de l'env de test
├── README.md                                 # Documentation principale du projet
└── pyproject.toml                            # Dépendances complètes (Dev/Test/Lint)
```

```text
┌──────────────────────────────────────────────────────────────────┐
│                         CLIENT (curl / UI)                       │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP
                             ▼
┌───────────────────────────────────────────────────────────────────┐
│                     FastAPI  (app/main.py)                        │
│  ┌────────────┐  ┌──────────────────┐   ┌──────────────────────┐  │
│  │ GET /health│  │   POST /ask      │   │   POST /rebuild      │  │
│  └────────────┘  └────────┬─────────┘   └──────────┬───────────┘  │
│                           │                        │ X-Rebuild-Key│
│              ┌────────────▼────────────┐           │              │
│              │  app/api/dependencies   │           │              │
│              │  get_rag(request)       │           │              │
│              └────────────┬────────────┘           │              │
└───────────────────────────┼────────────────────────┼──────────────┘
                            │ app.state.rag          │
                            ▼                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                    RAG Layer  (app/rag/)                         │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              EventRAGPipeline                             │   │
│  │  ┌─────────────────────┐   ┌───────────────────────────┐  │   │
│  │  │  MistralAIEmbeddings│   │     ChatMistralAI (LLM)   │  │   │
│  │  │  (mistral-embed)    │   │  (devstral-small-latest)  │  │   │
│  │  └──────────┬──────────┘   └───────────────────────────┘  │   │
│  │             │                          ▲                  │   │
│  │  ┌──────────▼──────────┐    LCEL chain │                  │   │
│  │  │    FAISS Index      ├───────────────┘                  │   │
│  │  │  (faiss-cpu)        │                                  │   │
│  │  └──────────┬──────────┘                                  │   │
│  └─────────────┼─────────────────────────────────────────────┘   │
└────────────────┼─────────────────────────────────────────────────┘
                 │ save / load
                 ▼
        data/faiss_index/
        ├── index.faiss
        └── index.pkl

┌──────────────────────────────────────────────────────────────────┐
│                    Data Layer  (app/data/)                       │
│   OpenAgendaFetcher  ──>  EventDocumentProcessor                 │
│   (httpx async)            (LangChain Documents)                 │
└──────────────────────────────────────────────────────────────────┘
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Construit avec

<!-- Le projet: -->

* [![GitHub][GitHub.com]][GitHub-url]
<!-- * [![Hugging Face Spaces][Hugging Face Spaces.com]][Hugging Face Spaces-url]
* [![GitHub Actions][GitHub Actions.com]][GitHub Actions-url] -->
* [![FastAPI][FastAPI.com]][FastAPI-url]
* [![Pydantic][Pydantic.com]][Pydantic-url]
<!-- * [![PostgreSQL][PostgreSQL.com]][PostgreSQL-url]
* [![Supabase][Supabase.com]][Supabase-url]
* [![SQLAlchemy][SQLAlchemy.com]][SQLAlchemy-url] -->
* [![Pytest-cov][Pytest-cov.com]][Pytest-cov-url]
* [![Swagger][Swagger.com]][Swagger-url]
* [![Docker][docker.com]][docker-url]
<!-- * [![Sphinx][Sphinx.com]][Sphinx-url] -->

<!-- Le modèle du projet 4:-->

* [![Python][Python.com]][Python-url]
* [![Poetry][Poetry.com]][Poetry-url]
* [![Jupyter][Jupyter.com]][Jupyter-url]
* [![Matplotlib][Matplotlib.com]][Matplotlib-url]
* [![NumPy][NumPy.com]][NumPy-url]
* [![Pandas][Pandas.com]][Pandas-url]
<!-- * [![Scikit-learn][Scikit-learn.com]][Scikit-learn-url] -->
<!-- * [![CatBoost][CatBoost.com]][CatBoost-url] -->

* [![LangChain][langchain.com]][langchain-url]
* [![FAISS][faiss.com]][faiss-url]
* [![Ragas][ragas.com]][ragas-url]
* [![Mistral][mistral.com]][mistral-url]
* [![Ruff][ruff.com]][ruff-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Suivez ces instructions pour installer le projet localement et lancer l'API sur votre machine de développement.

### Prerequis

<!-- #### Local -->

* Python 3.12
* Clé API Mistral: [lien_mistral](<https://admin.mistral.ai/organization/api-keys>)
* URL publique OpenAgenda: [lien_openagenda](<https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda/api/?disjunctive.keywords_fr&disjunctive.location_region&disjunctive.location_countrycode&disjunctive.location_department&disjunctive.location_city&dataChart=eyJxdWVyaWVzIjpbeyJjb25maWciOnsiZGF0YXNldCI6ImV2ZW5lbWVudHMtcHVibGljcy1vcGVuYWdlbmRhIiwib3B0aW9ucyI6eyJkaXNqdW5jdGl2ZS5rZXl3b3Jkc19mciI6dHJ1ZSwiZGlzanVuY3RpdmUubG9jYXRpb25fcmVnaW9uIjp0cnVlLCJkaXNqdW5jdGl2ZS5sb2NhdGlvbl9jb3VudHJ5Y29kZSI6dHJ1ZSwiZGlzanVuY3RpdmUubG9jYXRpb25fZGVwYXJ0bWVudCI6dHJ1ZSwiZGlzanVuY3RpdmUubG9jYXRpb25fY2l0eSI6dHJ1ZSwicmVmaW5lLnVwZGF0ZWRhdCI6IjIwMjUiLCJyZWZpbmUubG9jYXRpb25fY2l0eSI6IlBhcmlzIn19LCJjaGFydHMiOlt7ImFsaWduTW9udGgiOnRydWUsInR5cGUiOiJsaW5lIiwiZnVuYyI6IkFWRyIsInlBeGlzIjoiYWdlX21pbiIsInNjaWVudGlmaWNEaXNwbGF5Ijp0cnVlLCJjb2xvciI6IiNGRjUxNUEifV0sInhBeGlzIjoibGFzdGRhdGVfZW5kIiwibWF4cG9pbnRzIjoiIiwidGltZXNjYWxlIjoieWVhciIsInNvcnQiOiIifV0sImRpc3BsYXlMZWdlbmQiOnRydWUsImFsaWduTW9udGgiOnRydWV9&location=10,48.86138,2.63191&basemap=jawg.light>)

* **Pip ou Poetry**

```bash
# Vérifier la version python
python --version

# Si besoin d'installation (Debian/Ubuntu)
sudo apt update && sudo apt install python3.12 python3-pip

# Ou Poetry
# curl -sSL https://install.python-poetry.org | python3 -
# # Ajoutez Poetry à votre PATH si nécessaire, puis vérifiez
# poetry --version
```
<!-- 
* **Création d'une base PostgreSQL locale**

  Pour exécuter les tests d'intégration, vous devez disposer d'une instance PostgreSQL locale.

  1. *Installation* :

      ```bash
      sudo apt install postgresql postgresql-contrib
      ```

  2. *Accès au terminal psql* :

      ```bash
      sudo -u postgres psql
      ```

  3. *Initialisation de la DB* :

      ```sql
      CREATE DATABASE ml_test_db;
      CREATE USER test_user WITH PASSWORD 'votre_mot_de_passe';
      GRANT ALL PRIVILEGES ON DATABASE ml_test_db TO test_user;
      ```

#### Distant

* **Création d'un dépôt distant GitHub** :

    Créez un compte sur [GitHub](https://github.com/), créez un nouveau dépôt vide et connectez votre projet local :

    ```bash
    git remote add origin https://github.com/votre-user/votre-projet.git
    git push -u origin main
    ```

    Pour les secrets, allez dans Settings > Secrets and variables > Actions pour ajouter vos secrets (HUGGINGFACE_TOKEN, SB_HOST, etc.).

* **Création d'un espace sur Hugging Face**:

    Créez un compte sur Hugging Face, cliquez sur "New Space", choisissez le SDK Docker et un nom pour votre projet.Une fois l'espace créé, dans vos paramètres de profil, créez un "Write Token" pour permettre à GitHub de pousser le code.

    Concernant les secrets, allez dans les paramètres de votre Space, ajoutez les variables d'environnement de votre base de données Supabase pour que l'API puisse s'y connecter au runtime.

* **Création d'une base PostgreSQL Supabase**:

    Créez un compte et un projet sur Supabase puis cliquez sur le bouton Connect sur la barre de tâche supérieure à côté du nom de la base pour récupérez les informations de connexion. A noté que vous pourrez reset votre mot de passe de la base si celui-ci ne vous convient plus dans Project Settings > Database. -->

### Installation et configuration

#### Cloner le dépôt

```bash
git clone https://github.com/S-S-Zheng/Openclassrooms_P9.git
cd Openclassrooms_P9.git
```

#### Installer l'environnement

* **Poetry**:

  ```bash
    # Activer l'environnement virtuel
    python -m venv venv
    source venv/bin/activate

    # Poetry
    poetry init
    poetry install
    poetry shell

    # Configurer les variables d'environnement
    cp .env.example .env
    cp .env.example .env.test
    # Éditer .env et .env.test :
    # Variables (filtres...) ET MISTRAL_API_KEY, OPENAGENDA_PUBLIC_URL, REBUILD_API_KEY
    ```

<!-- #### Variables d'environnement

Créez un fichier .env et env.test à la racine pour configurer l'accès à L'espace de Hugging Face, votre base de données de test local (à créer) et distante Supabase. **TOUJOURS VERIFIER LEUR PRESENCE DANS .gitignore. CES FICHIERS NE DOIVENT JAMAIS APPARAITRE DANS LE DEPOT**

1. **.env**:

    ```python
    HUGGINGFACE_USERNAME = Nom_utilisateur_HF
    HUGGINGFACE_SPACE_NAME = Nom_espace_HF
    HUGGINGFACE_TOKEN = Token_HF

    SB_USER=Nom_utilisateur_SB
    SB_PASSWORD=Mot_de_passe_database_SB
    SB_HOST=Nom_hote_SB
    SB_PORT=Numero_port_SB
    SB_DB=Nom_database_SB
    ```

2. **.env.test**:

    ```python
    POSTGRES_USER=Nom_utilisateur_local
    POSTGRES_PASSWORD=Mot_de_passe_database_loca
    POSTGRES_HOST=Nom_hote_local
    POSTGRES_PORT=Numero_port_local
    POSTGRES_DB=Nom_database_local
    ``` -->

#### Test local

Pour démarrer l'API FastAPI localement avec rechargement automatique (Hot Reload):

  ```bash
  uvicorn app.main:app --reload --port 8000
  ```

L'API sera disponible sur : [http://localhost:8000](http://localhost:8000).

Avant de pousser vos modifications, vérifiez que l'ensemble de la suite de tests est au vert:

  ```bash
  # Lancer tous les tests avec rapport de couverture
  pytest
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

L'API expose plusieurs points d'entrée pour interagir avec le modèle de prédiction :

* **Poser une question** :
    Envoyez un POST sur /ask avec votre question "question" et le nombre de résultats "top_k" à sortir.
* **Reconstruire l'index (protégé)** :
    Envoyez un POST sur /rebuild avec la clef de protection X-Rebuild-Key afin de déclencher un ré-indexage. *Attention, l'indexation est une action qui peut être coûteuse et longue.*

<!-- * **Prédiction individuelle** :
  Envoyez un POST sur /predict/manual avec des features a remplir. Les features obligatoires sont détaillées dans l'exemple auquel on a ajouté deux features optionnelles supplémentaires.
* **Prédiction par paquet à partir d'un fichier de données** :
  Envoyez un POST sur /predict/upload avec le fichier de données.
* **Analyse des features** :
  Accédez à /feature-importance pour comprendre les facteurs clefs de la démission.
* **Rapport de profilage** :
  Accédez aux rapports de profilage afin d'étudier les goulots détranglement avec le nom du fichier (/reports/{filename} ou /reports/) ou le plus récent (/reports/ avec la case vide). (Attention, ne fonctionne qu'en local pour le moment)
* **Informations sur le modèle** :
  Accédez à /model-info/ pour avoir des informations sur le modèle. -->
* **Documentation interactive** :
  Une interface Swagger complète est disponible à la racine.

### Exemple de requête via cURL

<!-- ```text
curl -X 'POST' \
  'https://s254-p8-oc.hf.space/predict/manual' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "AMT_ANNUITY": 24700.5,
  "BUREAU_AMT_CREDIT_SUM_DEBT_MEAN": 100000,
  "DAYS_BIRTH": -9461,
  "DAYS_EMPLOYED": -637,
  "EXT_SOURCE_COUNT": 1,
  "NAME_FAMILY_STATUS": "Married",
  "OTHER_FEATURE_1": 0.5,
  "OTHER_FEATURE_400": 12,
  "OWN_CAR_AGE": 40,
  "PAYMENT_RATE": 0.1,
  "PREV_DAYS_LAST_DUE_1ST_VERSION_MAX": 0
}'
``` -->

```bash
curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{"question": "Des activités fun et gratuit ?", "top_k":4}'
```

```text
{"question":"Des activités fun et gratuit ?","answer":"Voici les activités fun et gratuites trouvées dans les documents :\n\n1. **Le rendez-vous des Makers** (17 janvier 2026, 12h00)\n   - Ateliers pratiques, démonstrations et échanges pour petits et grands.\n   - Lieu : Carrefour Numérique², Cité des Sciences (30 avenue Corentin Cariou, 75019 Paris).\n   - Accès : Métro ligne 7, station Porte de la Villette.\n   - Source : [Lien](http://www.cite-sciences.fr)\n\n2. **A la découverte des sciences !** (Certains mercredis, 10h00-12h00)\n   - Expériences de physique, conseils d'observation du ciel et jeux mathématiques.\n   - Lieu : Palais de la Découverte (Square Jean Perrin, 17 avenue Eisenhower, 75008 Paris).\n   - Accès : Entrée par le square Jean Perrin uniquement.\n   - Source : [Lien](http://palais-decouverte.fr)\n\n*Remarque* : Les autres événements mentionnés ne sont pas gratuits (webinaire en ligne) ou ne sont pas clairement indiqués comme tels.","sources":[{"content":"TITRE : Pour les enfants !\n--- CONTENU DÉTAILLÉ ---\nDétails : Cycle Enfance de l'Art\nAdresse : 100, rue Mouffetard, Paris\nVille : Paris\nPériode : Mercredi 21 janvier, 13h45\n","event_title":"Pour les enfants !","event_uid":"22354676","score":0.6257},{"content":"TITRE : Le rendez-vous des Makers\n--- CONTENU DÉTAILLÉ ---\nDétails : Rejoignez-nous pour une journée unique au Carrefour Numérique². Ce moment festif mettra à l’honneur l’inventivité et le partage, avec des makers passionnés prêts à vous étonner !✨ Au programme :\n\ndes ateliers pratiques,\ndes démonstrations captivantes\ndes échanges inspirants\npour petits et grands curieux\n\nL'accès est libre et gratuit. On vous attend pour une journée pleine de surprises et de créativité !\nAdresse : 30 avenue Core","event_title":"Le rendez-vous des Makers","event_uid":"27911821","score":0.625},{"content":"TITRE : Découvrez comment l'IA peut booster votre activité artisanale\n--- CONTENU DÉTAILLÉ ---\nDétails : Participez à notre webinaire gratuit le 19 janvier de 12h15 à 13h15 et découvrez, en seulement 1 heure, des usages concrets pour gagner du temps, améliorer votre visibilité et développer votre activité ; quel que soit votre niveau de maîtrise des outils numériques.\nAu programme :• Comprendre l’IA simplement, avec des exemples métiers,• Astuces pratiques : communication, relation client, organ","event_title":"Découvrez comment l'IA peut booster votre activité artisanale","event_uid":"92747157","score":0.6214},{"content":"TITRE : A la découverte des sciences !\n--- CONTENU DÉTAILLÉ ---\nDétails : Participez à des expériences de physique, recevez des conseils d'observation du ciel, et jouez avec les mathématiques!Tous les mercredis matin, entre 10h00 et 12h00, les médiateurs scientifiques du Palais de la découverte vous donnent rendez-vous au cœur de l'exposition.\nAdresse : Square Jean Perrin - 17 avenue Eisenhower, 75008 Paris\nVille : Paris\nTéléphone : 01 85 53 99 74\nWeb : http://palais-decouverte.fr\nPériode : 7 ja","event_title":"A la découverte des sciences !","event_uid":"97538686","score":0.6194}],"processing_time_ms":1663.0}(livrable-p9-py3.12)
```

### Documentation Interactive

Une fois l'API lancée, accédez à l'interface Swagger pour tester les endpoints en direct :

* **Local** :
  `http://localhost:8000/docs` (Swagger UI)
  `http://localhost:8000/redoc` (ReDoc)
<!-- * **Production** :
  `https://huggingface.co/spaces/Nom_utilisateur_HF/Nom_espace_HF` -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Deployment -->
<!-- ## Deploiement

Le déploiement est entièrement automatisé via une architecture MLOps :

* Hook pré-commit (filtre local):
  Le hook de pré-commit s'exécute automatiquement sur ta machine à chaque tentative de git commit. Son rôle est de s'assurer qu'aucun code "sale" ou mal formaté ne quitte ton poste de travail.

* CI (GitHub Actions) :
  Chaque modification poussée sur les branches main ou develop déclenche automatiquement le pipeline.
  Il est constitué de deux jobs:
  
  lint : Vérifie la conformité du code en passant par isort, black et flake8 (bien que le pré-commit le fasse déjà, la présence de ce job permet de s'assurer que le linting est effectuée)

  test : Ne se lance que si le linting est validé, ce job a pour fonction de tester le comportement unitaire et fonctionnel du code. Il lance un conteneur éphemère PostgreSQL 15 de test et execute pytest en suivant les directives du pytest.ini.

* CD (GitHub Actions) :
  Le déploiement ne se lance que si le pipeline de CI a réussi (workflow_run success). Il est restreint à la branche main pour garantir que seul le code de production est déployé.

  Au lieu de pousser tout le dépôt (ce qui serait lourd et risqué), le script crée un dossier éphémère hf_deploy. Il sélectionne les fichiers afin de garantir une conteneurisation optimisée; La seléction comprend le code de l'API, la logique ML et les artefacts associés (.cbm, .pkl). Cela garantit que les secrets locaux ou les données volumineuses (notebooks, png, csv ...) ne sont jamais exposés sur Hugging Face.

  Le script génère aussi dynamiquement un README.md avec un bloc YAML (frontmatter). C'est ce fichier qui configure Hugging Face (SDK Docker, port 8000, version Python, licence).

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->

<!-- Authentication -->
<!-- ## Authentification

L'accès à la base de données Supabase est sécurisé via des variables d'environnement unitaires (secrets) injectées au moment du runtime, garantissant qu'aucun identifiant ne circule en clair dans le dépôt. Les variables sont gérées via les GitHub Secrets et Hugging Face Secrets, isolant les secrets même pendant l'exécution du conteneur.

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->

## Démarrage avec Docker

```bash
# Construire et démarrer (potentiellement en sudo)
docker compose up --build

# Vérifier la santé
curl http://localhost:8000/health

# Construire l'index (première fois ou mise à jour)
curl -X POST http://localhost:8000/rebuild \
  -H "X-Rebuild-Key: Clef_rebuild"

# Poser une question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Poser une question...", "top_k":1}'
```

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->

<!-- Security -->
<!-- ## Securité

La sécurité des données d'entrée est assurée par une plusieurs couches de validation :

### Protection des Données et de la DB

* **Proxy de Données** :
  L'utilisateur n'interagit jamais directement avec la base de données ; l'API agit comme une passerelle sécurisée.
* **Connexion Chiffrée & Pooler** :
Utilisation d'un connection pooler avec chiffrement SSL pour toutes les communications avec Supabase.
* **Protection contre les Doublons** :
  Système de Hash unique (SHA-256) pour chaque prédiction, garantissant l'intégrité et servant de mécanisme de cache sécurisé.
* **Traçabilité (Auditing)** :
  Journalisation systématique de chaque requête dans une table de logs dédiée pour l'audit et la détection d'anomalies.

### Validation et Intégrité

* **Validation de Schéma (Pydantic)** :
  Filtrage strict des types de données en entrée de l'API.
* **Contrôle de Cohérence Métier** :
  Rejet automatique des données ne respectant pas la logique métier.

### Sécurité du Runtime (Hugging Face)

* **Gestion des Secrets** :
  Utilisation des "Repository Secrets" de GitHub et des "Space Secrets" de Hugging Face. Aucun identifiant n'est présent dans le code.
* **Isolation Docker** :
  Exécution dans un environnement sandboxé sans privilèges root.
* **Filtrage des Sorties** :
  Les exceptions internes sont interceptées pour ne jamais exposer la structure de la base de données ou du système dans les réponses HTTP.

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->

<!-- Evaluation -->
## Évaluation (RAGAS)

L'évaluation va permettre de juger de la pertinence d'un RAG sous différentes coutures en regardant sa capacité à rester dans le sujet, à répondre correctement et sur tous les aspects demandés et à pointer vers les bonnes sources.
En plus d'un index opérationnel, il est nécéssaire de se constituer un ensemble de questions/réponses (Q&A) afin de pouvoir jauger de la fiabilité du RAG.

### QA

```json
[
  {
    "question": "Je cherche un truc fun à faire pour occuper les gosses dans la rue Mouffetard (lieu et contact)?",
    "ground_truth": "'Pour les enfants !' pourrait vous interesser. C'est au 100, rue Mouffetard, Paris par contre aucun moyen de contact n'est mentionné."
  },
  {
    "question": "Trouve moi un evènement autour de l'IA où l'on peut discuter et débattre sur ce sujet, pas cher et pour le 13 janvier (lieu, prix)?",
    "ground_truth": "'[CaféIA #4] Découvrir l’IA' propose un espace d'écoute et de rencontre pour pouvoir débattre autour de l'IA. L'adresse de l'evènement est le 3 rue d'Aligre et l'entrée est gratuite"
  },
  {
    "question": "Retrouve moi les informations sur un webinar concernant l'appel à projet AMSC (nom, date, conditions).",
    "ground_truth": "'COFUND 2026 - Session d'information' est un webinar qui permet d'aider certaines structures à monter un programme de formation et de mobilité doctorale ou postdoctorale. Il se déroulera le 13 janvier 2026 à partir de 13h, en ligne et nécéssite de s'être inscrit au préalable."
  }
]
```

### Métriques utilisées

| Métrique | Définition | Jugement |
|----------|-----------|----------|
| **Faithfulness** | Les affirmations de la réponse sont-elles toutes ancrées dans le contexte récupéré ? | LLM |
| **AnswerRelevancy** | La réponse répond-elle bien à la question ? | LLM + Embedding |
| **ContextRecall** | Le contexte contient-il les informations nécessaires pour répondre ? | LLM |
| **ContextPrecision** | Les chunks récupérés sont-ils tous pertinents (pas de bruit) ? | LLM |

### Lancer l'évaluation

```bash
python -m CLI.eval_rag
# ou avec un fichier personnalisé :
python -m CLI.eval_rag --qa-file datas/raw/qa_pairs.json
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Technical Doc -->
##  Documentation technique

### Architecture du système

```text
┌──────────────────────────┐      ┌──────────────────────────┐      ┌─────────────────────────┐
│   Source (OpenAgenda)    │ ───> │   ETL (Python/YAML)      │ ───> │  Vector DB (FAISS)      │
└──────────────────────────┘      └──────────────────────────┘      └─────────────────────────┘
             ▲                                                            │
             │                    ┌──────────────────────────┐            │ Context
             └─────────────────── │    LLM (Mistral AI)      │ <──────────┘
               Generated Answer   └──────────────────────────┘
```

### Préparation et indexation

* **Extraction** : Utilisation d'un OpenAgendaFetcher asynchrone pour récupérer les données JSON.
* **Nettoyage** : Filtrage des événements sans contexte, uid ou titre. On exclu en plus les eléments pré-2026 et hors Paris, Île-de-France (on place aussi des filtres du type fr, horloge de Paris et commencement de l'évènement ascendant).
* **Chunking** : On réalise un chunking de 1000 caractères avec un chevauchement de 200 caractères. Utilisation d'un mapping YAML pour structurer le texte envoyé à l'embedding : TITRE + DESCRIPTION + CONTEXTE.
* **Embedding** : Utilisation de mistral-embed (dimension 2048 avec une température de 0.1). Logique de batching pour respecter les limites de l'API.

### Choix du modèle NLP

* **Modèle sélectionné** : devstral-small-latest.
* **Pourquoi ?** :
  * **Langue** : Optimisé pour le Français (convient bien à OpenAgenda).
  * **Coût/Perf** : Fenêtre de contexte très elevée et leger à déployer. Bonne performance sur l'arrangement contextuelle bien que ces modèles ne soient pas adapté pour le Q&A général.
  * **Fiabilité** : Très bon respect des consignes de "ne pas inventer" (hallucinations réduites).
* **Prompting** : Utilisation d'un System Prompt strict définissant l'identité de l'assistant et l'obligation de citer les sources via l'event_uid.

### Base vectorielle (FAISS)

* **Type** : FAISS indexé par similarité cosinus.
* **Persistance** : Sauvegarde binaire dans data/faiss_index/.
* **Contexte et métadonnées** : Conservation d'un bon nombre d'informations permettant une bonne restitution du contexte ainsi qu'un filtrage de qualité.

### Exemple de résultats

----------- Scores ------------

| Metrique                       |   Score |    Qualité |
|--------------------------------|---------|------------|
| answer_relevancy               |  0.7902 | Acceptable |
| context_precision              |  0.8333 | Acceptable |
| context_recall                 |  0.6667 |        Bad |
| faithfulness                   |  0.8095 | Acceptable |

```json
{
    "scores": {
        "answer_relevancy": 0.7901574909557875,
        "faithfulness": 0.8095238095238094,
        "context_recall": 0.6666666666666666,
        "context_precision": 0.83333333325
    },
    "qa": [
        {
            "question": "Je cherche un truc fun \u00e0 faire pour occuper les gosses dans la rue Mouffetard (lieu et contact)?",
            "ground_truth": "'Pour les enfants !' pourrait vous interesser. C'est au 100, rue Mouffetard, Paris par contre aucun moyen de contact n'est mentionn\u00e9.",
            "answer": "Pour occuper les enfants dans la rue Mouffetard, vous pouvez les emmener au **Cycle Enfance de l'Art** (\u00e9v\u00e9nement \"Pour les enfants !\"), situ\u00e9 au **100, rue Mouffetard, Paris**.\n\nAucun contact sp\u00e9cifique n'est mentionn\u00e9 dans les documents fournis.",
            "n_contexts": 5
        },
        {
            "question": "Trouve moi un ev\u00e8nement autour de l'IA o\u00f9 l'on peut discuter et d\u00e9battre sur ce sujet, pas cher et pour le 13 janvier (lieu, prix)?",
            "ground_truth": "'[Caf\u00e9IA #4] D\u00e9couvrir l\u2019IA' propose un espace d'\u00e9coute et de rencontre pour pouvoir d\u00e9battre autour de l'IA. L'adresse de l'ev\u00e8nement est le 3 rue d'Aligre et l'entr\u00e9e est gratuite",
            "answer": "L'\u00e9v\u00e9nement \"[Caf\u00e9IA #4] D\u00e9couvrir l\u2019IA\" correspond \u00e0 votre recherche :\n- **Lieu** : 3, rue Aligre, Paris\n- **Prix** : Entr\u00e9e libre / gratuit\n- **Format** : Informel, ouvert \u00e0 tous, favorisant les \u00e9changes et d\u00e9bats sur l'IA.\n\n*Source* : [Caf\u00e9IA #4](https://openagenda.com/francenum/events/cafe-ia-decouvrir-lia)\n\n*Note* : L'\u00e9v\u00e9nement du 13 janvier 2026 (R\u00e9sultat #3) est en anglais et sur inscription, mais ne mentionne pas de d\u00e9bat.",
            "n_contexts": 5
        },
        {
            "question": "Retrouve moi les informations sur un webinar concernant l'appel \u00e0 projet AMSC (nom, date, conditions).",
            "ground_truth": "'COFUND 2026 - Session d'information' est un webinar qui permet d'aider certaines structures \u00e0 monter un programme de formation et de mobilit\u00e9 doctorale ou postdoctorale. Il se d\u00e9roulera le 13 janvier 2026 \u00e0 partir de 13h, en ligne et n\u00e9c\u00e9ssite de s'\u00eatre inscrit au pr\u00e9alable.",
            "answer": "Le webinaire concernant l'appel \u00e0 projets AMSC (Actions Marie Sk\u0142odowska-Curie) COFUND 2026 s'intitule **\"COFUND 2026 - Session d'information\"**. Il se d\u00e9roulera le **13 janvier 2026 en ligne**.\n\n**Conditions et d\u00e9tails** :\n- L'appel \u00e0 projets a \u00e9t\u00e9 ouvert le **16 d\u00e9cembre 2025**.\n- Date limite de soumission des propositions : **8 avril 2026**.\n- Au programme : pr\u00e9sentation de l'action COFUND, statistiques des pr\u00e9c\u00e9dents appels, retour d'exp\u00e9rience d'un laur\u00e9at, conseils d'un expert, et session de questions-r\u00e9ponses.\n- Anim\u00e9 par un intervenant de la Commission europ\u00e9enne et le Point de Contact National (PCN) AMSC.\n\nSource : [OpenAgenda](https://openagenda.com/agenda-de-leurope-en-guyane/events/cofund-2026-session-dinformation).",
            "n_contexts": 5
        }
    ]
}
```

* **answer_relevancy de 0.79** -> La réponse est pertinente par rapport à la question.
* **faithfulness de 0.81** -> La réponse est bien basée sur les docs.
* **context_recall de 0.67** -> Parfois, l'index manque de détails pour répondre à 100%.
* **context_precision de0.83** -> Les documents remontés sont très pertinents.

> Ces valeurs sont indicatives ; les scores dépendent du corpus indexé
> et des questions du jeu de test.

<!-- La documentation détaillée des modules Python est générée via Sphinx et est disponible dans le dossier docs/build/html et générable via:

  ```bash
  # Générer la documentation Sphinx
  cd docs
  make html
  ```

Un pdf "documentation_technique.pdf" est fourni avec dans docs/. -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Improvements -->
## Pistes d'amélioration

### Qualité RAG

1. **Reranking** : Ajouter un reranking, un check par le LLM, un threshold après le retrieval FAISS pour trier plus finement les résultats de FAISS.
2. **HyDE** (Hypothetical Document Embeddings) : Générer une réponse hypothétique
   avant l'embedding de la question pour améliorer le recall.
3. **Gestion filtres** : Améliorer les métadonnées et données contexte afin d'améliorer le RAG.
4. **Parent-child chunking** : Indexer les petits chunks, retourner les documents
   parents pour plus de contexte à la génération.

### Infrastructure

1. **Streaming** : Implémenter `StreamingResponse` sur `/ask` via `chain.astream()` pour une interface plus réactive.
2. **Cache** : Redis pour les questions fréquentes (TTL 1 h).
3. **Index incrémental** : Ne ré-embedder que les nouveaux événements plutôt que tout reconstruire pour économiser les tokens d'embedding.
4. **Observabilité** : Intégrer LangSmith ou Langfuse pour le tracing des chaînes LangChain.
5. **Gunicorn** : Remplacer uvicorn mono-worker par `gunicorn -k uvicorn.workers.UvicornWorker`
   pour la prod multi-cœur.
6. **Tests de charge** : Locust ou k6 pour valider le comportement sous charge.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP
## Roadmap

* [x] Création d'un dépôt Github pour le projet
* [x] Création d'une classe qui regroupe les méthodes de ML
* [x] Mise en place de tests - part 1
* [x] Automatisation CI/CD (GitHub Actions) - part 1
* [x] Création de l'API avec FastAPI
* [x] Mise en place de tests - part 2
* [x] Automatisation CI/CD (GitHub Actions) - part 2
* [x] Mise en place de la base de données PostgreSQL (Supabase)
* [x] Mise en place de tests - part 3
* [x] Automatisation CI/CD (GitHub Actions) - part 3
* [x] Documentation Sphinx autogénérée

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->

<!-- CONTRIBUTING
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors

<a href="https://github.com/github_username/repo_name/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=github_username/repo_name" alt="contrib.rocks image" />
</a>
-->

<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - <email@email_client.com>

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

ACKNOWLEDGMENTS
## Acknowledgments

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->

<!-- =============================================================================== -->
<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

<!-- [product-screenshot]: images/Gemini_P5.png -->

<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->
<!--
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/

[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/

[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/

[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/

[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/

[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com

[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com

[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com

[Gradio.com]: https://img.shields.io/badge/Gradio-F97316?logo=Gradio&logoColor=white
[Gradio-url]: https://www.gradio.app/

[OpenAPI.com]: https://img.shields.io/badge/OpenAPI-6BA539?logo=openapiinitiative&logoColor=white
[OpenAPI-url]: https://swagger.io/specification/

[MkDocs.com]: https://img.shields.io/badge/MkDocs-526CFE?logo=materialformkdocs&logoColor=fff
[MkDocs-url]: https://www.mkdocs.org/
-->

<!--My list of badges-->
[GitHub.com]: https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white
[GitHub-url]: https://github.com/

[FastAPI.com]: https://img.shields.io/badge/FastAPI-009485.svg?logo=fastapi&logoColor=white
[FastAPI-url]: https://fastapi.tiangolo.com/

[Pydantic.com]: https://img.shields.io/badge/Pydantic-E92063?logo=Pydantic&logoColor=white
[Pydantic-url]: https://docs.pydantic.dev/latest/

[Pytest-cov.com]: https://img.shields.io/badge/Pytest--cov-%233F51B5?style=for-the-badge&logo=pytest&logoColor=white&labelColor=black
[Pytest-cov-url]: https://pypi.org/project/pytest-cov/

[Swagger.com]: https://img.shields.io/badge/Swagger-85EA2D?logo=swagger&logoColor=173647
[Swagger-url]: https://swagger.io/

<!-- [Sphinx.com]: https://img.shields.io/badge/Sphinx-000?logo=sphinx&logoColor=fff
[Sphinx-url]: https://www.sphinx-doc.org/en/master/ -->

<!--Utilisés pour le modèle-->
[Python.com]: https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff
[Python-url]: https://www.python.org/

[Poetry.com]: https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json
[Poetry-url]: https://python-poetry.org/

[Jupyter.com]: https://img.shields.io/badge/Jupyter-ffffff?logo=JupyterB
[Jupyter-url]: https://jupyter.org/

[Matplotlib.com]: https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff
[Matplotlib-url]: https://matplotlib.org/

[NumPy.com]: https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff
[NumPy-url]: https://numpy.org/

[Pandas.com]: https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff
[Pandas-url]: https://pandas.pydata.org/docs/index.html

<!-- [CatBoost.com]: https://img.shields.io/badge/CatBoost-FF4632?logo=catboost&logoColor=white
[CatBoost-url]: https://catboost.ai/docs/en/ -->

[langchain.com]: https://img.shields.io/badge/LangChain-1c3c3c.svg?logo=langchain&logoColor=white
[langchain-url]: https://www.langchain.com/

[faiss.com]: XXX
[faiss-url]: https://faiss.ai/index.html

[ragas.com]: https://docs.ragas.io/en/stable/_static/imgs/ragas-logo.png
[ragas-url]: https://docs.ragas.io/en/stable/getstarted/

[docker.com]: https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=fff
[docker-url]: https://www.docker.com/

[mistral.com]: https://img.shields.io/badge/Mistral%20AI-FA520F?logo=mistral-ai&logoColor=fff
[mistral-url]: https://docs.mistral.ai/

[ruff.com]: https://custom-icon-badges.demolab.com/badge/Ruff-261230.svg?logo=ruff-logo
[ruff-url]: https://docs.astral.sh/ruff/
