from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    value = json.loads(text) if path.suffix.lower() == ".json" else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if isinstance(value, dict):
        value = value.get("answers", [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain a list of answer records")
    return value


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(left, str) and isinstance(right, str):
        return left.strip().casefold() == right.strip().casefold()
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_equal(item_left, item_right)
            for item_left, item_right in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return (
            set(left) == set(right)
            and all(_values_equal(left[key], right[key]) for key in left)
        )
    return left == right


def evaluate_answers(
    predicted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predicted_by_id = {record["id"]: record for record in predicted if "id" in record}
    expected_by_id = {record["id"]: record for record in expected if "id" in record}
    wrong: list[dict[str, Any]] = []
    true_positive = false_positive = false_negative = 0

    for question_id, expected_record in expected_by_id.items():
        predicted_record = predicted_by_id.get(question_id)
        is_correct = (
            predicted_record is not None
            and _values_equal(predicted_record.get("answer"), expected_record.get("answer"))
        )
        if is_correct:
            true_positive += 1
        else:
            false_negative += 1
            if predicted_record is not None:
                false_positive += 1
            wrong.append(
                {
                    "id": question_id,
                    "pipeline_output": predicted_record,
                    "correct_answer": expected_record,
                }
            )

    extra_predictions = set(predicted_by_id) - set(expected_by_id)
    false_positive += len(extra_predictions)
    for question_id in sorted(extra_predictions):
        wrong.append(
            {
                "id": question_id,
                "pipeline_output": predicted_by_id[question_id],
                "correct_answer": None,
            }
        )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    performance = {
        "evaluated": len(expected_by_id),
        "correct": true_positive,
        "wrong": len(wrong),
        "extra_predictions": len(extra_predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return performance, wrong


def _build_run_command(args: argparse.Namespace, pipeline_output: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_questions.py")),
        str(args.questions),
        str(args.data_root),
        str(pipeline_output),
        "--planner",
        args.planner,
    ]
    for option, value in (
        ("--rulebook", args.rulebook),
        ("--calendar", args.calendar),
        ("--llm-provider", args.llm_provider),
        ("--llm-base-url", args.llm_base_url),
        ("--llm-model", args.llm_model),
        ("--llm-api-key", args.llm_api_key),
        ("--llm-timeout", args.llm_timeout),
        ("--llm-request-delay", args.llm_request_delay),
    ):
        if value is not None:
            command.extend((option, str(value)))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run questions and evaluate answers against a reference file"
    )
    parser.add_argument("questions", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument(
        "--answers",
        type=Path,
        default=Path("train/train/answers.jsonl"),
        help="Reference answers JSONL or JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Output root passed to run_questions.py (default: .\\outputs)",
    )
    parser.add_argument(
        "--planner", choices=("heuristic", "llm", "qwen"), default="llm"
    )
    parser.add_argument("--rulebook", type=Path)
    parser.add_argument("--calendar", type=Path)
    parser.add_argument("--llm-provider")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-timeout", type=float)
    parser.add_argument("--llm-request-delay", type=float)
    args = parser.parse_args()

    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(_build_run_command(args, output_root), check=True)

    answers_dir = output_root / f"{args.questions.stem}_answers"
    generated_answers = answers_dir / "answers.json"
    predicted = _read_records(generated_answers)
    question_ids = {
        record["id"]
        for record in _read_records(args.questions)
        if "id" in record
    }
    expected = [
        record
        for record in _read_records(args.answers)
        if record.get("id") in question_ids
    ]
    predicted = [
        record
        for record in predicted
        if record.get("id") in question_ids
    ]
    performance, wrong = evaluate_answers(predicted, expected)
    (answers_dir / "performance.json").write_text(
        json.dumps(performance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (answers_dir / "wrong_answers").open("w", encoding="utf-8") as handle:
        for record in wrong:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(performance, ensure_ascii=False))


if __name__ == "__main__":
    main()
