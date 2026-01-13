"""Run reproducible evaluations against prepared task datasets.

This script loads the JSONL files produced by the task-specific
`scripts/create_datasets.py` helpers (expected under the repo-level
`/datasets` directory by default), queries a chosen model via the shared
`models` package, and reports exact-match accuracy plus aggregated token
usage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.base import ensure_usage_aliases, normalize_messages  # type: ignore
from models.factory import create_model  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark reproduction over prepared task datasets"
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        default=DATASETS_DIR,
        help="Path to directory containing *_datasets.jsonl files (default: /datasets at repo root)",
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="+",
        default=None,
        help="Specific JSONL dataset files to load (overrides glob). Relative paths are resolved against --datasets.",
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["api", "openai", "deepseek", "vllm"],
        help="Model backend to use (api/openai/deepseek/vllm)",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model identifier or path (passed to shared models factory)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Optional batch size hint for vLLM (default: 1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples per dataset file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write aggregated results as JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample progress",
    )
    parser.add_argument(
        "--save-response",
        type=Path,
        default=None,
        help="Optional path to save prompts/responses as JSONL for all samples",
    )
    return parser.parse_args()


ANSWER_REGEX = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def extract_answer(response: str) -> str:
    """Strictly extract text inside <answer> tags; missing tags -> empty string."""
    if response is None:
        return ""
    match = ANSWER_REGEX.search(response)
    if not match:
        return ""
    return match.group(1).strip()


def load_jsonl(path: Path, limit: int | None = None) -> List[Dict]:
    samples: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def evaluate_dataset(
    model,
    samples: List[Dict],
    verbose: bool = False,
    response_writer=None,
) -> Tuple[float, Dict[str, float]]:
    total = len(samples)
    exact = 0
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}

    for idx, sample in enumerate(samples):
        question = sample.get("question")
        expected = (sample.get("answer") or "").strip()

        prompt_input = normalize_messages(question) if isinstance(question, list) else question
        response, usage = model.generate(prompt_input)
        usage_norm = ensure_usage_aliases(usage)

        prediction = extract_answer(response).strip().lower()
        expected_norm = expected.strip().lower()
        if prediction == expected_norm:
            exact += 1

        for key in usage_totals:
            usage_totals[key] += usage_norm.get(key, 0) or 0

        if verbose:
            print(f"[{idx+1}/{total}] EM={'Y' if prediction == expected_norm else 'N'} | prompt_tokens={usage_norm.get('prompt_tokens',0)}")
            print("PROMPT:", prompt_input)
            print("RESPONSE:", response)

        if response_writer is not None:
            response_writer.write(
                json.dumps(
                    {
                        "index": idx,
                        "prompt": prompt_input,
                        "response": response,
                        "expected": expected,
                        "usage": usage_norm,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    exact_match_rate = exact / total if total else 0.0
    return exact_match_rate, usage_totals


def main() -> None:
    args = parse_args()
    datasets_dir = args.datasets if args.datasets.is_absolute() else (REPO_ROOT / args.datasets)

    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    dataset_files: List[Path] = []
    if args.files:
        for file_path in args.files:
            candidate = file_path if file_path.is_absolute() else datasets_dir / file_path
            if not candidate.exists():
                raise FileNotFoundError(f"Dataset file not found: {candidate}")
            dataset_files.append(candidate)
    else:
        dataset_files = sorted(datasets_dir.glob("*_datasets.jsonl"))

    if not dataset_files:
        raise FileNotFoundError(f"No *_datasets.jsonl files found in {datasets_dir}")

    model_config = {"temperature": 0.0, "max_tokens": 1024, "batch_size": args.batch_size}
    model = create_model(args.model_type, args.model_name, config=model_config)

    results: Dict[str, Dict] = {}

    for path in dataset_files:
        samples = load_jsonl(path, limit=args.max_samples)

        response_writer = None
        if args.save_response:
            args.save_response.parent.mkdir(parents=True, exist_ok=True)
            response_writer = args.save_response.open("a", encoding="utf-8")

        em_rate, usage_totals = evaluate_dataset(
            model, samples, verbose=args.verbose, response_writer=response_writer
        )

        if response_writer:
            response_writer.close()

        results[path.stem] = {
            "num_samples": len(samples),
            "exact_match_rate": em_rate,
            "token_usage": usage_totals,
        }

        print(f"{path.name}: EM={em_rate:.3f} (n={len(samples)}) | tokens={usage_totals}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
