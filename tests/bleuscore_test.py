import pytest
from unittest.mock import patch, MagicMock
import torch
from audiolm.functional import compute_bleu


# Mock Tokenizer
class MockTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        # simulates decoded output with translation token for ids 0 and 1, and no token for id 2
        mapping = {
            0: "<translation> Die Katze sitzt auf dem Tisch",
            1: "<translation> The dog runs in the park",
            2: "some other task output",
        }
        return mapping.get(ids[0].item(), "") # Only look at first token (task token)


@pytest.fixture
def mock_dataset():
    return [
        {"English": "The cat is on the table", "German": "Die Katze sitzt auf dem Tisch"},
        {"German": "Der Hund läuft im Park", "English": "The dog runs in the park"},
    ]


@pytest.fixture
def output_ids():
    return torch.tensor([[0], [1], [2]]) # Tensor simulating model outputs [3, 1]


@pytest.fixture
def input_texts():
    return [
        "The cat is on the table",
        "Der Hund läuft im Park",
        "some other input",
    ]


def test_compute_bleu_filters_translation_token(mock_dataset, output_ids, input_texts):
    """Only samples with translation token should be evaluated."""
    with patch("audiolm.functional.load_dataset", return_value=mock_dataset): # replace dataset with mock dataset
        score = compute_bleu(
            output_ids=output_ids,
            input_texts=input_texts,
            tokenizer=MockTokenizer(),
        )

    assert score > 0.0


def test_compute_bleu_no_translation_tokens(mock_dataset, input_texts):
    """If no sample has translation token, score should be 0."""
    output_ids = torch.tensor([[2], [2], [2]])

    with patch("audiolm.functional.load_dataset", return_value=mock_dataset):
        score = compute_bleu(
            output_ids=output_ids,
            input_texts=input_texts,
            tokenizer=MockTokenizer(),
        )
    assert score == 0.0


def test_compute_bleu_returns_float(mock_dataset, output_ids, input_texts):
    """Return type should be float."""
    with patch("audiolm.functional.load_dataset", return_value=mock_dataset):
        score = compute_bleu(
            output_ids=output_ids,
            input_texts=input_texts,
            tokenizer=MockTokenizer(),
        )
    assert isinstance(score, float)


def test_compute_bleu_perfect_match(mock_dataset, input_texts):
    """Check whether bleu score calculation works when translations are valid."""
    class PerfectTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            mapping = {
                0: "<translation> Die Katze sitzt auf dem Tisch",
                1: "<translation> The dog runs in the park",
            }
            return mapping.get(ids[0].item(), "")

    output_ids = torch.tensor([[0], [1]])

    with patch("audiolm.functional.load_dataset", return_value=mock_dataset):
        score = compute_bleu(
            output_ids=output_ids,
            input_texts=input_texts[:2],
            tokenizer=PerfectTokenizer(),
        )
    assert score > 50.0