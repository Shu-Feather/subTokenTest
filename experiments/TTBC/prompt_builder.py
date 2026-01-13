"""
Utility to build TTBC-ready datasets from existing task prompt builders.

For each supported task, we reuse the task's native prompt construction code when possible,
and fall back to lightweight formatting when only minimal fields are available. The output
is a JSON list of objects with at least:
  - prompt: string to send to the model
  - answer: reference answer when available (may be None)
  - meta: task-specific metadata preserved from the source sample
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cli.py").exists():
            return parent
    return Path(__file__).resolve().parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Task-local imports (guarded to avoid hard failures when a task is unused)
try:
    from tasks.aligned_table.src.prompt_builder import PromptBuilder as AlignedTablePromptBuilder
except Exception:
    AlignedTablePromptBuilder = None

try:
    from tasks.cipher_decipher.src.utils.prompts import PromptTemplates as CipherPromptTemplates
    from tasks.cipher_decipher.src.utils.enums import TaskType as CipherTaskType
except Exception:
    CipherPromptTemplates = None
    CipherTaskType = None

try:
    from tasks.map_navigation.prompts import get_sokoban_prompt, get_frozenlake_prompt
except Exception:
    get_sokoban_prompt = None
    get_frozenlake_prompt = None

try:
    from tasks.rsa_randomart.src.utils import create_prompt as rsa_create_prompt
except Exception:
    rsa_create_prompt = None

try:
    from tasks.context_aware_redaction.src.benchmark import ContextAwareRedactionBenchmark
except Exception:
    ContextAwareRedactionBenchmark = None
try:
    from tasks.biological_sequence.src.prompt_templates import PromptTemplates as BioPromptTemplates
except Exception:
    BioPromptTemplates = None


def load_records(path: str) -> List[Dict[str, Any]]:
    """Load JSON/JSONL and flatten common container keys."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if p.suffix.lower() == ".jsonl":
        raw: Any = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    else:
        raw = json.loads(p.read_text())

    def _flatten(item: Any) -> List[Dict[str, Any]]:
        if isinstance(item, list):
            return item
        if isinstance(item, dict):
            for key in ("samples", "data", "test_cases"):
                if key in item and isinstance(item[key], list):
                    return item[key]
            if "datasets" in item and isinstance(item["datasets"], dict):
                out: List[Dict[str, Any]] = []
                for val in item["datasets"].values():
                    out.extend(val)
                return out
            # RSA / gomoku style: wrap as single element
            return [item]
        raise ValueError("Unsupported dataset format")

    return _flatten(raw)


@dataclass
class TTBCSample:
    prompt: str
    answer: Optional[str]
    meta: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"prompt": self.prompt, "answer": self.answer, "meta": self.meta}


class BaseAdapter:
    def build(self, records: Sequence[Dict[str, Any]]) -> List[TTBCSample]:
        raise NotImplementedError


class TreeAdapter(BaseAdapter):
    def build(self, records: Sequence[Dict[str, Any]]) -> List[TTBCSample]:
        out: List[TTBCSample] = []
        for rec in records:
            prompt = rec.get("prompt") or rec.get("question") or ""
            answer = rec.get("expected_answer")
            out.append(TTBCSample(prompt=prompt, answer=answer, meta=rec))
        return out


class ContextRedactionAdapter(BaseAdapter):
    def __init__(self):
        self.prompt_builder = None
        if ContextAwareRedactionBenchmark:
            try:
                # Minimal config to drive prompt formatting only
                cfg = {"prompt": {"restricted_reasoning": False}}
                self.prompt_builder = ContextAwareRedactionBenchmark(config=cfg, model=None, verbose=False)
            except Exception:
                self.prompt_builder = None

    @staticmethod
    def create_prompt(text: str, restricted_reasoning: bool = False) -> str:
        prompt = f"""You are a data privacy assistant. Your task is to redact (mask) sensitive information in the given text according to specific rules.

**Redaction Rules:**

1. **18-digit ID Card Number**: Keep the first 6 digits and last 2 digits, mask the middle 10 digits with asterisks (*)
   - Example: 123456789012345678 → 123456**********78

2. **Phone Number** (format: +[area code] [11-digit number]): Keep the '+' sign and area code, keep the first 3 digits and last 4 digits of the phone number, mask the middle 4 digits with asterisks (*)
   - Example: +12 12345678901 → +12 123****8901

3. **Credit Card Number**: Keep the first 6 digits and last 4 digits, mask all middle digits with asterisks (*)
   - Example: 1234 5678 9012 3456 → 123456******3456

**Instructions:**
- Carefully identify all sensitive information in the text (phone numbers, ID card numbers, credit card numbers)
- Apply the appropriate masking rule for each type of sensitive information
- Keep all other text exactly the same
- Place your final redacted text between <answer> and </answer> tags

**Text to redact:**
{text}

**Your redacted text (place between <answer></answer> tags):**"""
        if restricted_reasoning:
            prompt += (
                "\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
            )
        return prompt

    def build(self, records: Sequence[Dict[str, Any]]) -> List[TTBCSample]:
        out: List[TTBCSample] = []
        for rec in records:
            text = rec.get("original_context") or rec.get("text") or ""
            answer = rec.get("redacted_context")
            restricted_reasoning = rec.get("restricted_reasoning", False)
            prompt = None
            if self.prompt_builder:
                try:
                    # Update restricted flag if present
                    self.prompt_builder.restricted_reasoning = restricted_reasoning
                    prompt = self.prompt_builder.create_prompt(text)
                except Exception:
                    prompt = None
            if prompt is None:
                prompt = self.create_prompt(text, restricted_reasoning=restricted_reasoning)
            out.append(TTBCSample(prompt=prompt, answer=answer, meta=rec))
        return out


class AlignedTableAdapter(BaseAdapter):
    def __init__(self):
        self.builder = AlignedTablePromptBuilder() if AlignedTablePromptBuilder else None

    def build(self, records: Sequence[Dict[str, Any]]) -> List[TTBCSample]:
        out: List[TTBCSample] = []
        for rec in records:
            if self.builder:
                # Default to text format when missing
                table_format = rec.get("table_format") or "text"
                test_case = dict(rec)
                test_case["table_format"] = table_format
                prompt = self.builder.build_prompt(test_case)
            else:
                prompt = rec.get("context") or rec.get("description") or ""
            out.append(TTBCSample(prompt=prompt, answer=None, meta=rec))
        return out


class MapNavigationAdapter(BaseAdapter):
    def build(self, records: Sequence[Dict[str, Any]]) -> List[TTBCSample]:
        out: List[TTBCSample] = []
        for rec in records:
            env_type = (rec.get("env_type") or rec.get("environment") or "").lower()
            map_str = rec.get("map") or rec.get("board") or ""
            question = rec.get("question") or ""
            prompt: str
            if env_type == "sokoban" and get_sokoban_prompt:
                prompt = get_sokoban_prompt(map_str, question)
            elif env_type == "frozenlake" and get_frozenlake_prompt:
                prompt = get_frozenlake_prompt(map_str, question)
            else:
                prompt = f"Environment:\n{map_str}\n\nQuestion: {question}\nAnswer:"
            answer = rec.get("answer")
            out.append(TTBCSample(prompt=prompt, answer=answer, meta=rec))
        return out


class BioSeqAdapter(BaseAdapter):
    def build(self, records: Sequence[Dict[str, Any]]) -> List[TTBCSample]:
        out: List[TTBCSample] = []
        for rec in records:
            task = rec.get("task_type") or rec.get("task") or "bio_seq"
            seq_in = rec.get("input") or rec.get("sequence") or ""
            expected = rec.get("expected_output") or rec.get("output")
            prompt = None
            if BioPromptTemplates:
                try:
                    prompt = BioPromptTemplates.get_prompt_for_task(task, seq_in, restricted_reasoning=False)
                except Exception:
                    prompt = None
            if prompt is None:
                prompt = (
                    f"You are a biology assistant. Task: {task}. Input sequence: {seq_in}\n"
                    "Provide the correct output sequence inside <answer>...</answer>."
                )
            out.append(TTBCSample(prompt=prompt, answer=expected, meta=rec))
        return out


class CipherAdapter(BaseAdapter):
    def __init__(self):
        self.templates = CipherPromptTemplates() if CipherPromptTemplates else None

    def build(self, records: Sequence[Dict[str, Any]]) -> List[TTBCSample]:
        out: List[TTBCSample] = []
        for rec in records:
            task_type = rec.get("task_type")
            text = rec.get("input") or rec.get("text") or ""
            shift = rec.get("shift")
            style = rec.get("prompt_style") or "basic"
            prompt = None
            if self.templates and task_type and CipherTaskType:
                ttype = CipherTaskType(task_type)
                if ttype == CipherTaskType.MORSE_ENCODE:
                    prompt = self.templates.get_morse_encode_prompt(text, style, restricted_reasoning=False)
                elif ttype == CipherTaskType.MORSE_DECODE:
                    prompt = self.templates.get_morse_decode_prompt(text, style, restricted_reasoning=False)
                elif ttype == CipherTaskType.CAESAR_ENCODE:
                    prompt = self.templates.get_caesar_encode_prompt(text, shift, style, restricted_reasoning=False)
                elif ttype == CipherTaskType.CAESAR_DECODE:
                    prompt = self.templates.get_caesar_decode_prompt(text, shift, style, restricted_reasoning=False)
            if prompt is None:
                prompt = f"Task: {task_type}\nInput: {text}\nProvide answer:"
            answer = rec.get("output") or rec.get("expected_output")
            out.append(TTBCSample(prompt=prompt, answer=answer, meta=rec))
        return out


class RSAAdapter(BaseAdapter):
    def build(self, records: Sequence[Dict[str, Any]]) -> List[TTBCSample]:
        out: List[TTBCSample] = []
        for rec in records:
            p1 = rec.get("pattern1") or []
            p2 = rec.get("pattern2") or []
            expected = rec.get("ground_truth")
            if isinstance(p1, str):
                p1_lines = p1.splitlines()
            else:
                p1_lines = list(p1)
            if isinstance(p2, str):
                p2_lines = p2.splitlines()
            else:
                p2_lines = list(p2)
            prompt = None
            if rsa_create_prompt:
                try:
                    prompt = rsa_create_prompt(p1_lines, p2_lines, restricted_reasoning=False)
                except Exception:
                    prompt = None
            if prompt is None:
                prompt = (
                    "You are given two RSA key fingerprint patterns. Find all differences.\n\n"
                    f"Pattern 1:\n{''.join(p1_lines)}\n\nPattern 2:\n{''.join(p2_lines)}\n\n"
                    "List coordinate differences as (x, y): old -> new within <answer> tags."
                )
            out.append(TTBCSample(prompt=prompt, answer=expected, meta=rec))
        return out


class GenericAdapter(BaseAdapter):
    def build(self, records: Sequence[Dict[str, Any]]) -> List[TTBCSample]:
        out: List[TTBCSample] = []
        for rec in records:
            prompt = rec.get("prompt") or rec.get("question") or rec.get("input") or json.dumps(rec)
            answer = rec.get("answer") or rec.get("expected") or rec.get("label")
            out.append(TTBCSample(prompt=prompt, answer=answer, meta=rec))
        return out


ADAPTERS: Dict[str, BaseAdapter] = {
    "tree": TreeAdapter(),
    "context_aware_redaction": ContextRedactionAdapter(),
    "aligned_table": AlignedTableAdapter(),
    "map_navigation": MapNavigationAdapter(),
    "biological_sequence": BioSeqAdapter(),
    "cipher_decipher": CipherAdapter(),
    "rsa_randomart": RSAAdapter(),
    # Fallback for remaining tasks (adversarial_prompt, gomoku, typewriter, etc.)
    "generic": GenericAdapter(),
}


def select_adapter(task_type: str) -> BaseAdapter:
    return ADAPTERS.get(task_type, ADAPTERS["generic"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TTBC-ready prompts from task datasets.")
    parser.add_argument("--task_type", required=True, help="Task name (e.g., tree, aligned_table, map_navigation)")
    parser.add_argument("--input", required=True, help="Path to source dataset (JSON or JSONL)")
    parser.add_argument("--output", required=False, help="Where to save the TTBC dataset (JSON). Prints to stdout if omitted.")
    args = parser.parse_args()

    records = load_records(args.input)
    adapter = select_adapter(args.task_type)
    samples = adapter.build(records)

    payload = [s.as_dict() for s in samples]
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {len(samples)} prompts to {out_path}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
