from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import unicodedata
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


def _aliases_for(value: Any) -> list[Any]:
    """Normalize one answer's accepted aliases to a list."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _answer_matches(predicted: Any, expected_record: dict[str, Any]) -> bool:
    """Compare an answer with the canonical value and its reference aliases.

    List answers use a parallel alias list: each position contains the
    alternatives accepted for the corresponding canonical list item. Table
    aliases are aligned with expected rows.
    """
    expected = expected_record.get("answer")
    aliases = expected_record.get("alias")

    is_table = expected_record.get("type") == "table" or (
        isinstance(expected, dict) and "columns" in expected and "rows" in expected
    )
    if is_table:
        return _table_answers_match(predicted, expected, aliases)

    if isinstance(expected, list):
        if not isinstance(predicted, list) or len(predicted) != len(expected):
            return False
        aligned_aliases = aliases if isinstance(aliases, list) else []
        return all(
            _value_matches_aliases(
                actual,
                canonical,
                _aliases_for(aligned_aliases[index])
                if index < len(aligned_aliases)
                else [],
            )
            for index, (actual, canonical) in enumerate(zip(predicted, expected))
        )

    return _value_matches_aliases(predicted, expected, _aliases_for(aliases))


def _value_matches_aliases(
    predicted: Any, expected: Any, aliases: list[Any]
) -> bool:
    return _values_equal(predicted, expected) or any(
        _values_equal(predicted, alias) for alias in aliases
    )


def _normalized_column_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold()).replace("\u0111", "d")
    return "".join(character for character in decomposed if character.isalnum())


def _table_answers_match(
    predicted: Any, expected: Any, aliases: Any
) -> bool:
    """Compare table columns by name when possible and rows as an unordered set."""
    if not isinstance(predicted, dict) or not isinstance(expected, dict):
        return False
    predicted_columns = predicted.get("columns")
    expected_columns = expected.get("columns")
    predicted_rows = predicted.get("rows")
    expected_rows = expected.get("rows")
    if not all(
        isinstance(value, list)
        for value in (predicted_columns, expected_columns, predicted_rows, expected_rows)
    ):
        return False
    if (
        len(predicted_columns) != len(expected_columns)
        or len(predicted_rows) != len(expected_rows)
        or not all(isinstance(name, str) for name in predicted_columns + expected_columns)
    ):
        return False

    column_count = len(expected_columns)
    if not all(
        isinstance(row, list) and len(row) == column_count
        for row in predicted_rows + expected_rows
    ):
        return False

    # If all labels map uniquely, reorder predicted cells to reference order.
    # Otherwise compare columns positionally as the fallback contract specifies.
    normalized_predicted = [_normalized_column_name(name) for name in predicted_columns]
    normalized_expected = [_normalized_column_name(name) for name in expected_columns]
    if (
        len(set(normalized_predicted)) == column_count
        and set(normalized_predicted) == set(normalized_expected)
        and len(set(normalized_expected)) == column_count
    ):
        predicted_positions = {
            name: index for index, name in enumerate(normalized_predicted)
        }
        positions = [
            predicted_positions[name] for name in normalized_expected
        ]
    else:
        positions = list(range(column_count))

    aligned_predicted_rows = [
        [row[index] for index in positions] for row in predicted_rows
    ]
    row_aliases = aliases if isinstance(aliases, list) else []

    def row_matches(actual: list[Any], canonical: list[Any], index: int) -> bool:
        alternatives = (
            _aliases_for(row_aliases[index])
            if index < len(row_aliases)
            else []
        )
        alias_column = next(
            (i for i, value in enumerate(canonical) if isinstance(value, str)),
            None,
        )
        return all(
            _value_matches_aliases(
                actual_value,
                expected_value,
                alternatives if cell_index == alias_column else [],
            )
            for cell_index, (actual_value, expected_value) in enumerate(
                zip(actual, canonical)
            )
        )

    # Use a maximum matching so duplicate or aliased rows cannot be reused.
    matched_expected: dict[int, int] = {}

    def assign(expected_index: int, visited: set[int]) -> bool:
        for predicted_index, actual in enumerate(aligned_predicted_rows):
            if predicted_index in visited or not row_matches(
                actual, expected_rows[expected_index], expected_index
            ):
                continue
            visited.add(predicted_index)
            previous = matched_expected.get(predicted_index)
            if previous is None or assign(previous, visited):
                matched_expected[predicted_index] = expected_index
                return True
        return False

    return all(assign(index, set()) for index in range(len(expected_rows)))


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
            and _answer_matches(predicted_record.get("answer"), expected_record)
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
        "--output",
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
        help=(
            "Reference answers JSONL or JSON file (default: answers.jsonl in "
            "the parent directory of data_root)"
        ),
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

    if args.answers is None:
        args.answers = args.data_root.parent / "answers.jsonl"
    if not args.answers.is_file():
        parser.error(f"Reference answers file not found: {args.answers}")

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
    if question_ids and not expected:
        raise SystemExit(
            f"No reference answer IDs from {args.questions} were found in {args.answers}"
        )

    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(_build_run_command(args, output_root), check=True)

    answers_dir = output_root / f"{args.questions.stem}_answers"
    generated_answers = answers_dir / "answers.jsonl"
    predicted = _read_records(generated_answers)
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
