from __future__ import annotations

import argparse
from pathlib import Path

from datamate.pipeline import answer_questions
from datamate.planning.llm import LLMParser
from datamate.planning.heuristic import HeuristicParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer DataMate JSONL questions")
    parser.add_argument("questions", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument(
        "output", type=Path, nargs="?", default=Path("outputs"),
        help="Output directory (default: .\\outputs) or legacy JSONL output path",
    )
    parser.add_argument("--schema-output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--rulebook", type=Path)
    parser.add_argument("--calendar", type=Path)
    parser.add_argument(
        "--evidence-dir", type=Path,
        help="Directory for per-question pipeline evidence bundles",
    )
    parser.add_argument(
        "--planner", choices=("heuristic", "llm", "qwen"), default="llm",
        help=(
            "Planner; the LLM planner uses --llm-provider or LLM_PROVIDER "
            "(local llama-server Gemma GGUF by default)"
        ),
    )
    parser.add_argument(
        "--llm-provider",
        choices=("local", "gemma4-local", "openrouter", "huggingface", "groq", "gemini"),
        help="OpenAI-compatible endpoint preset (default: LLM_PROVIDER or local)",
    )
    parser.add_argument("--llm-base-url", help="Override the provider API base URL")
    parser.add_argument("--llm-model", help="Override the provider model name")
    parser.add_argument("--llm-api-key", help="Override the provider API key")
    parser.add_argument("--llm-timeout", type=float, help="Request timeout in seconds")
    parser.add_argument(
        "--llm-request-delay",
        type=float,
        help="Minimum seconds between consecutive LLM requests (default: 1; or LLM_REQUEST_DELAY)",
    )
    args = parser.parse_args()
    output_root = args.output
    if output_root.suffix.lower() == ".jsonl":
        output_path = output_root
        evidence_dir = args.evidence_dir
        answers_root = output_root.parent
    else:
        answers_root = output_root / f"{args.questions.stem}_answers"
        output_path = answers_root / "answers.json"
        evidence_dir = args.evidence_dir or answers_root
    answer_questions(
        args.questions,
        args.data_root,
        output_path,
        parser=(
            HeuristicParser()
            if args.planner == "heuristic"
            else LLMParser(
                provider=args.llm_provider,
                base_url=args.llm_base_url,
                model=args.llm_model,
                api_key=args.llm_api_key,
                timeout=args.llm_timeout,
                request_delay=args.llm_request_delay,
            )
        ),
        schema_output_path=args.schema_output,
        trace_output_path=args.trace_output,
        rulebook_path=args.rulebook,
        calendar_path=args.calendar,
        evidence_output_dir=evidence_dir,
        token_usage_output_path=(
            answers_root / "token_usage.jsonl"
            if output_root.suffix.lower() != ".jsonl"
            else None
        ),
    )


if __name__ == "__main__":
    main()
