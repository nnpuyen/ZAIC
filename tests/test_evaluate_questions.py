from __future__ import annotations

import json
import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).parents[1] / "evaluate_questions.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_questions", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
evaluate_answers = _MODULE.evaluate_answers


def test_evaluate_answers_writes_standard_exact_match_counts() -> None:
    performance, wrong = evaluate_answers(
        [
            {"id": "correct", "answer": 10.0},
            {"id": "wrong", "answer": 2},
        ],
        [
            {"id": "correct", "answer": 10},
            {"id": "wrong", "answer": 3},
            {"id": "missing", "answer": 4},
        ],
    )

    assert performance == {
        "evaluated": 3,
        "correct": 1,
        "wrong": 2,
        "extra_predictions": 0,
        "precision": 0.5,
        "recall": 1 / 3,
        "f1": 0.4,
    }
    assert json.loads(json.dumps(wrong)) == [
        {
            "id": "wrong",
            "pipeline_output": {"id": "wrong", "answer": 2},
            "correct_answer": {"id": "wrong", "answer": 3},
        },
        {
            "id": "missing",
            "pipeline_output": None,
            "correct_answer": {"id": "missing", "answer": 4},
        },
    ]
