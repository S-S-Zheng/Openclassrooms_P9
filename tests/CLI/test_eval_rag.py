"""
tests/test_eval_rag.py

On vérifie que le lancement de la CLI d'évaluation réalise bien la préparation de la donnée pour
évaluation ainsi que le print, save et la pipeline d'éxécution (main). On évite ragas_eval
car trop complexe:

``load_qa_pairs`` -> ``run_rag_on_qa`` -> ``prepare_for_ragas``
--> ``print_scores`` -> ``save_results``
puis ``main()``

"""

# imports
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from CLI.eval_rag import (
    load_qa_pairs,
    main,
    prepare_for_ragas,
    print_scores,
    run_rag_on_qa,
    save_results,
)

# =====================================================================


# ---------------- Vérifie le chargement des paires QA
@pytest.mark.unit
def test_load_qa_pairs_success(tmp_path):
    # Créer un faux fichier JSON
    fake_qa_file = tmp_path / "qa_pairs.json"
    fake_qa_file.write_text('[{"question": "Où ?", "ground_truth": "Ici"}]', encoding="utf-8")

    pairs = load_qa_pairs(tmp_path)

    # Assertions
    assert len(pairs) == 1
    assert pairs[0]["question"] == "Où ?"


# ---------------- Vérifie qu'on intègre bien ``"answer"`` et ``"contexts"``
@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_rag_on_qa_logic(fake_llm_response):
    """Vérifie qu'on intègre bien ``"answer"`` et ``"contexts"``"""
    # Mock du RAG
    mock_rag = MagicMock()
    mock_rag.query = AsyncMock(
        return_value={
            "answer": fake_llm_response,
            "source_documents": [MagicMock(page_content="Contexte 1")],
        }
    )
    qa_pairs = [{"question": "Où ?", "ground_truth": "Ici"}]

    results = await run_rag_on_qa(mock_rag, qa_pairs)

    # Assertions
    assert len(results) == 1
    assert results[0]["answer"] == "..."
    assert results[0]["contexts"] == ["Contexte 1"]


# ---------------- Vérifie la préparation du document pour ``ragas.evaluate()``
@pytest.mark.unit
def test_prepare_for_ragas_mapping(fake_llm_response):
    """Vérifie la préparation du document pour ``ragas.evaluate()``"""
    fake_full_result = [
        {
            "question": "Où ?",
            "answer": fake_llm_response,
            "contexts": ["Contexte 1", "Contexte 11"],
            "ground_truth": "Ici",
        }
    ]
    dataset = prepare_for_ragas(fake_full_result)

    # On vérifie que Ragas a bien reçu les données au bon endroit
    sample = dataset[0]

    # Assertions
    assert sample.user_input == "Où ?"
    assert sample.response == "..."  # type:ignore Pylance rejette le typage
    assert sample.retrieved_contexts == ["Contexte 1", "Contexte 11"]  # type:ignore


# ---------------- Vérifie que le terminal affiche bien les scores
@pytest.mark.unit
def test_print_scores_output(capsys):
    """Vérifie que ``print_scores()`` renvoi correctement le scoring"""
    # fake le scoring
    fake_scores = {"answer_relevancy": 0.9, "faithfulness": 0.5}

    print_scores(fake_scores)

    captured = capsys.readouterr()  # fixture natif de pytest permettant de lire le terminal

    # Assertions
    assert "Excellent" in captured.out  # Car 0.9 >= 0.85
    assert "Bad" in captured.out  # Car 0.5 < 0.7
    assert "answer_relevancy" in captured.out


# ---------------- Vérifie la logique de sauvegarde ``save_results()``
@pytest.mark.unit
def test_save_results_payload(tmp_path, fake_llm_response):
    """Vérifie la logique de sauvegarde ``save_results()``"""
    scores = {"faithfulness": 0.8}
    fake_full_result = [
        {
            "question": "Où ?",
            "answer": fake_llm_response,
            "contexts": ["Contexte 1", "Contexte 11"],
            "ground_truth": "Ici",
        }
    ]

    with patch("CLI.eval_rag.save_datas") as mock_save:
        save_results(scores, fake_full_result, tmp_path)
        # On vérifie que les données envoyées à save_datas sont correctes
        args, kwargs = mock_save.call_args
        payload = args[0]

        # Assertions
        assert payload["scores"]["faithfulness"] == 0.8
        assert payload["qa"][0]["n_contexts"] == 2


# ---------------- Vérifie la pipeline d'évaluation
@pytest.mark.asyncio
async def test_main_flow_success(fake_settings):
    with (
        patch("CLI.eval_rag.get_settings", return_value=fake_settings),
        patch("CLI.eval_rag.EventRAGPipeline") as MockRAG,
        patch("CLI.eval_rag.load_qa_pairs", return_value=[]),
        patch("CLI.eval_rag.run_rag_on_qa", return_value=[]),
        patch("CLI.eval_rag.prepare_for_ragas"),
        patch("CLI.eval_rag.ragas_eval", return_value={}),
        patch("CLI.eval_rag.print_scores"),
        patch("CLI.eval_rag.save_results"),
    ):
        # On simule un index chargé avec succès
        MockRAG.return_value.load_index = AsyncMock(return_value=True)

        await main(Path("fake_path"))

        # Assertions
        # On vérifie juste que la chaîne s'est déroulée
        MockRAG.return_value.load_index.assert_called_once()
