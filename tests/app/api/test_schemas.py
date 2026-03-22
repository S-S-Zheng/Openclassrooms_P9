"""
tests/app/api/test_schemas.py

Vérifie la validation et le nettoyage des données d'entrée Pydantic.
"""

import pytest
from pydantic import ValidationError

from app.api.schemas import AskRequest

# ========================================================================


# ==================== AskRequest ======================
@pytest.mark.unit
class TestAskRequest:
    # ------------------ Vérifie le stripping des espace autour de la question
    def test_ask_request_clean_question_strip(self):
        """Vérifie que les espaces blancs autour sont bien supprimés (val.strip())."""
        # Donnée brute avec espaces, tabulations et retours à la ligne
        raw_question = "   Est-ce qu'il y a du jazz ? \n\t "

        request = AskRequest(question=raw_question, top_k=5)

        # On vérifie que le validateur a bien nettoyé la chaîne
        assert request.question == "Est-ce qu'il y a du jazz ?"

    # ------------------ Vérifie que la longueur de la question est bien respectée
    @pytest.mark.parametrize(
        "question_length, expected_err",
        [
            ("ab", "at least 3"),  # Trop court
            ("a" * 1001, "at most 1000"),  # Trop long
        ],
    )
    def test_ask_request_min_length(self, question_length, expected_err):
        """Vérifie que la contrainte min_length=3 est respectée."""
        with pytest.raises(ValidationError) as exc:
            AskRequest(question=question_length)

        # On vérifie que l'erreur mentionne bien la longueur minimale
        assert expected_err in str(exc.value)

    # ------------------ Vérifier que le top_k est bien respectée
    @pytest.mark.parametrize("k_limit", [0, 11])
    def test_ask_request_top_k_limits(self, k_limit):
        """Vérifie que top_k est bien bridé entre 1 et 10 (ge=1, le=10)."""
        with pytest.raises(ValidationError):
            AskRequest(question="Est-ce qu'il y a du jazz ?", top_k=k_limit)
