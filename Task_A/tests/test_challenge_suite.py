from collections import Counter

from scripts.evaluate_challenges import EXPECTED_CATEGORIES, load_suite, summarize


def test_challenge_suite_has_required_coverage() -> None:
    suite = load_suite()
    examples = suite["examples"]
    assert len(examples) == 28
    assert Counter(example["category"] for example in examples) == {
        category: 4 for category in EXPECTED_CATEGORIES
    }


def test_challenge_suite_is_binary_and_self_describing() -> None:
    suite = load_suite()
    assert suite["suite_id"] == "binary-sentiment-challenge-v1"
    for example in suite["examples"]:
        assert example["expected_sentiment"] in {"negative", "positive"}
        assert example["rationale"].strip()
        assert example["text"].strip()


def test_challenge_summary_reports_failures_and_categories() -> None:
    rows = [
        {
            "category": "sarcasm",
            "expected_sentiment": "negative",
            "predicted_sentiment": "negative",
            "correct": True,
            "confidence": 0.8,
            "truncated": False,
        },
        {
            "category": "sarcasm",
            "expected_sentiment": "positive",
            "predicted_sentiment": "negative",
            "correct": False,
            "confidence": 0.9,
            "truncated": False,
        },
    ]
    summary = summarize(rows)
    assert summary["diagnostic_accuracy"] == 0.5
    assert summary["category_metrics"]["sarcasm"]["correct"] == 1
    assert len(summary["failures"]) == 1
