"""
scripts/eval_rag.py

Script CLI autonome qui évalue le système RAG via le framework RAGAS.

Métriques employées:

* ``Faithfulness``: Juge de la fidélité de la réponse par rapport aux documents
    (factuel ou hallucination?)
* ``ContextPrecision``: Juge le classement des sources (est-ce que le premier document est
    effectivement plus pertinent que le second...?)
* ``AnswerRelevancy``: Juge de la pertinence de la réponse (répond directement à tous les points
    demandées dans la question ou pas?)
* ``ContextRecall``: Juge de la fiabilité des sources (est-ce que les informations demandées
    dans la ground_truth sont bien présentes dans les documents?)

On a fini par s'abstenir d'évaluer "par défaut" (regarde toutes les métriques adaptées) pour
réduire le coût:
    Très couteuse: ``ContextPrecision`` et ``Faithfulness``. Elles demandent au LLM d'analyser
        chaque phrase de la réponse et de la comparer à chaque document source.
        Avec un nombre de contexte important, le nombre de comparaisons explose.
    Moyennement coûteuse: ``AnswerRelevancy``. Elle demande au LLM de générer plusieurs
        questions artificielles à partir de la ground_truth, puis de faire des calculs d'embeddings.
        Si le coût token est moins importante, le coût temporelle l'est de façon exponentielle.
    Les moins coûteuses: ``ContextRecall``. C'est une comparaison directe entre la ground_truth
    et les contextes. C'est plus rapide car la cible est fixe.
"""

# Imports
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
import json
import numpy as np
from openai import AsyncOpenAI # On utilise le client OpenAI pour la compatibilité avec ragas

# Ajout du dossier racine au PATH
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ragas.cache import DiskCacheBackend
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness
)
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

from app.core.config import get_settings, Settings
from app.rag.rag_pipeline import EventRAGPipeline
from app.utils.save_load_datas import save_datas


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_QA_PATH = _PROJECT_ROOT / "datas" / "raw"
_RESULTS_PATH = _PROJECT_ROOT / "datas" / "ragas"




# =========================== Chargement des QA =====================================


def load_qa_pairs(qa_path: Path, qa_filename: str = "qa_pairs.json") -> list[dict]:
    """
    Charge les QA annotés à partir d'un fichier.

    Parameters
    ----------
    qa_path (Path)
        Chemin contenant le fichier avec les QA annotés.
    qa_filename (str)
        Nom du fichier QA -> ``{"question": ..., "ground_truth": ...}``. defaut: "qa_pairs.json"

    Returns
    -------
    list[dict]
        Pairs de QA chargés

    Raises
    ------
    SystemExit
        Si le fichier est manquant, corrompu ou mal entré
    """
    path = qa_path / qa_filename
    if not path.exists():
        logger.error(f"Fichier de test QA introuvable: {path}")
        sys.exit(1)

    # pairs, _ = load_datas(path)
    # if isinstance(pairs, pd.DataFrame):
    #     pairs = pairs.to_dict(orient="records")
    # On n'utilise pas load_datas car renvoi un dataframe pour le JSON et pas envie de s'ajouter
    # des étapes pour rien...
    with open(path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
    

    if not isinstance(pairs, list) or not pairs:
        logger.error("Le fichier QA doit être un JSON array.")
        sys.exit(1)

    logger.info(f" {len(pairs)} paires de QA ont été chargé depuis {path}")
    return pairs


# =========================== Inférence RAG avec QA ===============================


async def run_rag_on_qa(
    rag: EventRAGPipeline,
    qa_pairs: list[dict],
) -> list[dict]:
    """
    Exécute le RAG sur chaque question pour collecter les réponses et contextes.

    Parameters
    ----------
    rag:
        Instance ``EventRAGPipeline``
    qa_pairs:
        Liste de QA ``{"question": ..., "ground_truth": ...}`` sous forme dict.

    Returns
    -------
    list[dict]
        Meme dicos avec en plus les cles ``"answer"`` et ``"contexts"``.
    """
    full_result: list[dict] = []

    for i, pair in enumerate(qa_pairs, start=1):
        question: str = pair["question"]
        logger.info(f"[{i}/{len(qa_pairs)}] Interrogation du RAG: {question[:60]}")
        try:
            result = await rag.query(question=question)
            answer: str = result["answer"]
            contexts: list[str] = [doc.page_content for doc in result["source_documents"]]
        except Exception as exc:
            logger.warning(f"Requete RAG echouée concernant '{question}':\n{exc}")
            answer = ""
            contexts = []

        full_result.append(
            {
                **pair,
                "answer": answer,
                "contexts": contexts,
            }
        )

    return full_result


# ========================== PREPARATION POUR EVAL ====================


def prepare_for_ragas(full_result: list[dict]) -> EvaluationDataset:
    """
    Convert paires QA enrichies avec ``answer`` et ``context`` en RAGAS ``EvaluationDataset``.

    Parameters
    ----------
    full_result:
        QA dicts avec ``question``, ``ground_truth``, ``answer``, ``contexts``.

    Returns
    -------
    EvaluationDataset
        Objet prêt pour l'évaluation ragas via ``ragas.evaluate()``.
    """
    # SingleTurnSample est est le format standard de RAGAS 0.2+.
    # On y met l'entrée utilisateur, la réponse de l'IA, les contextes extraits
    # de FAISS et la réponse de référence (ground truth).
    samples = [
        SingleTurnSample(
            user_input=pair["question"],
            retrieved_contexts=pair["contexts"],
            response=pair["answer"],
            reference=pair.get("ground_truth", ""),
        )
        for pair in full_result
    ]
    return EvaluationDataset(samples=samples) # type:ignore Pylance rejette le typage


# ========================== EVALUATION RAGAS ============================


async def ragas_eval(dataset: EvaluationDataset, settings: Settings) -> dict:
    """
    Lance une évaluation RAGAS sur les métriques:
        * ``Faithfulness``: La réponse est-elle basée sur les documents fournis ?
        * ``AnswerRelevancy``: La réponse répond-elle directement à la question ?

    Parameters
    ----------
    dataset (EvaluationDataset) : Objet contenant les samples (question, contexte, réponse, target)
    settings (Settings) : Configuration (clés API, noms des modèles)

    Returns
    -------
    dict : Dictionnaire contenant les scores moyens par métrique.
    """
    # Initialisation du cache et du client Mistral
    cache_ragas = DiskCacheBackend()

    # Configuration du client compatible Mistral
    # On utilise le client OpenAI pointant vers l'URL de Mistral
    # Détournement OBLIGATOIRE (si on veut rester sur Mistral a minima)
    client = AsyncOpenAI(
        base_url=settings.mistral_base_url,
        api_key=settings.mistral_api_key
    )

    # Utilisation des factories pour obtenir des objets InstructorLLM modernes
    judge_llm = llm_factory(
        model=settings.llm_model,
        client=client,
        cache=cache_ragas
    )
    judge_embeddings = embedding_factory(
        model=settings.embed_model,
        client=client,
        cache=cache_ragas
    )

    # Initialisation manuelle des Scorers
    metrics = {
        "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings), # type:ignore
        "faithfulness": Faithfulness(llm=judge_llm),
        "context_recall": ContextRecall(llm=judge_llm),
        "context_precision": ContextPrecision(llm=judge_llm)
    }
    results = {name: [] for name in metrics}
    # mapping des arguments des métriques car ragas prise de tete...
    arg_requirements = {
        "faithfulness": ["user_input", "response", "retrieved_contexts"],
        "answer_relevancy": ["user_input", "response"],
        "context_recall": ["user_input", "reference", "retrieved_contexts"],
        "context_precision": ["user_input", "reference", "retrieved_contexts"],
    }

    logger.info(f"Évaluation RAGAS avec {settings.llm_model} comme évaluateur...")


    for i, qa in enumerate(dataset):
        logger.info(f"Question {i+1}/{len(dataset)}...")
        # On itère sur métriques
        for name, scorer in metrics.items():
            try:
                # On extrait les args de dataset via le mapping
                kwargs = {arg: getattr(qa, arg) for arg in arg_requirements[name]}
                score = await scorer.ascore(**kwargs)
                # On s'assure de bien récupérer seulement la valeur de scoring et pas un objet Score
                # pour cela on va chercher dans 'value' si c'est une classe sinon ok
                results[name].append(float(getattr(score, 'value', score)))
            except Exception as e:
                # Pour éviter que la requête coupe pour X raison sur la enième question, on place
                # un nan si ça plante et on passe au suivant
                logger.error(f"Erreur {name} [Index {i}]: {e}")
                results[name].append(np.nan)

        # Pause pour respecter le Rate Limit de l'API Mistral
        if i < len(dataset) - 1:
            await asyncio.sleep(2.0)

    return {
        name: float(np.nanmean(scores))
        if not np.all(np.isnan(scores))
        else 0.0
        for name, scores in results.items()
    }


# ===================== RAPPORT =======================================

def print_scores(scores: dict[str, float]) -> None:
    """
    Print les score d'évaluation

    Parameters
    ----------
    scores:
        Metric name -> score (0–1) mapping.
    """
    print("\n----------- Scores ------------\n")
    print(f"| {'Metrique':<30} | {'Score':>7} | {'Qualité':>10} |")
    print(f"|{'-'*32}|{'-'*9}|{'-'*12}|")
    for name, score in sorted(scores.items()):
        quality = "Excellent" if score >= 0.85 else ("Acceptable" if score >= 0.7 else "Bad")
        print(f"| {name:<30} | {score:>7.4f} | {quality:>10} |")
    print()


def save_results(
    scores: dict,
    full_result: list[dict],
    result_path: Path,
    result_filename: str = "eval_results"
) -> None:
    """
    Sauvegarde les résultats d'évaluation et détails au format JSON.

    Parameters
    ----------
    scores (dict)
        Aggrège les scores
    full_result (list[dict])
        Les QA enrichies qui ont été utilisé.
    result_path (Path)
        Chemin de sauvegarde
    result_filename (str)
        Nom du fichier sauvegardé
    """
    # output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "scores": scores,
        "qa": [
            {
                "question": qa["question"],
                "ground_truth": qa.get("ground_truth", ""),
                "answer": qa.get("answer", ""),
                "n_contexts": len(qa.get("contexts", [])),
            }
            for qa in full_result
        ],
    }
    # with output_path.open("w", encoding="utf-8") as fh:
    #     json.dump(payload, fh, ensure_ascii=False, indent=2)
    save_datas(payload,result_path,filename=result_filename,format="json")
    logger.info(f"Resultats sauvegardés dans {result_path}")


# =========================================================
# ====================== TEST =============================
# =========================================================


async def main(qa_file: Path) -> None:
    """
    Pipeline d'évaluation: Charge -> infere rag -> évalue -> rapport

    Parameters
    ----------
    qa_file:
        Chemin du fichier QA annoté
    """
    # ========================= Initialisation =========================
    settings = get_settings()
    start = time.monotonic()
    # Charger l'index pré-construit
    rag = EventRAGPipeline(settings)
    if not await rag.load_index():
        logger.error("Index FAISS introuvable.")
        sys.exit(1)

    # ====================== Chargement des paires QA ===================
    qa_pairs = load_qa_pairs(qa_file)
    # ===================== Inférence rag ===========================
    full_result = await run_rag_on_qa(rag, qa_pairs)
    # ==================== Preparation des réultats pour éval ==================
    dataset = prepare_for_ragas(full_result)
    scores = await ragas_eval(dataset, settings)
    # ================== Print score et rapport ======================
    print_scores(scores)
    save_results(scores, full_result, _RESULTS_PATH)
    duration = time.monotonic() - start
    logger.info(f"Évaluation terminée en {duration:.2f} s ===")


# ======================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalue le RAG en utilisant des métriques RAGAS")
    parser.add_argument(
        "--qa-file",
        type=Path,
        default=_DEFAULT_QA_PATH,
        help="Chemin fu fichier QA annoté (defaut: datas/raw/qa_pairs.json).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.qa_file))
