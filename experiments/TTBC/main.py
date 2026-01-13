"""
TTBC Experiments (Test-Time Budget Control)

Implementation based on paper methodology:
- TTBC1 baseline (right of star): Iteratively replace </think> with "\nWait" for n_wait times
- TTBC1 budget control (left of star): Force exact thinking tokens by truncation or Wait injection
- TTBC2: Enforce exact thinking token budget t_exact with truncation or Wait injection

Key implementation details:
- During THINK phase, suppress the end-of-thinking delimiter </think>
- Strip delimiter by token ids for robustness across vLLM versions
- No step_limit - generate freely until </think> or model limit
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. Think step by step inside <think>...</think> "
    "and then provide the final answer inside <answer>...</answer>."
)

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_THINK_BUDGET = 32000
DEFAULT_ANSWER_MAX = 2048

# Append token "Wait" to the thinking trace
WAIT_TEXT = "\nWait"


@dataclass
class SampleResult:
    prompt: str
    expected: Optional[str]
    answer: str
    extracted_answer: str
    thinking_tokens: int
    answer_tokens: int
    thought: str
    n_wait: Optional[int] = None
    max_think_tokens: Optional[int] = None
    t_exact: Optional[int] = None
    finish_reason: Optional[str] = None  # "stop" | "budget_forced" | "exact"
    accuracy: Optional[bool] = None


@dataclass
class DetailedResponse:
    """Stores detailed prompt/response information for each generation round."""
    user_prompt: str
    expected_answer: Optional[str]
    full_thinking_prompt: str
    thinking_response: str
    full_answer_prompt: str
    answer_response: str
    extracted_answer: str
    thinking_tokens: int
    answer_tokens: int
    n_wait: Optional[int] = None
    max_think_tokens: Optional[int] = None
    t_exact: Optional[int] = None
    finish_reason: Optional[str] = None
    accuracy: Optional[bool] = None
    thinking_rounds: List[Dict[str, Any]] = field(default_factory=list)


def extract_answer(raw: str) -> str:
    """Extract text strictly inside <answer>...</answer>; return empty string if missing."""
    start_tag, end_tag = "<answer>", "</answer>"
    start = raw.find(start_tag)
    end = raw.find(end_tag, start + len(start_tag)) if start != -1 else -1
    if start != -1 and end != -1 and end > start:
        return raw[start + len(start_tag) : end].strip()
    return ""


def normalize(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def load_dataset(path: Optional[str]) -> List[Dict[str, Optional[str]]]:
    """Load JSON or JSONL dataset. If absent, return a one-sample sanity set."""
    if not path:
        return [{"prompt": "How many letter r are in the word 'raspberry'?", "answer": "3"}]

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if p.suffix.lower() == ".jsonl":
        data = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    else:
        data = json.loads(p.read_text())

    records: List[Dict[str, Optional[str]]] = []
    for item in data:
        prompt = item.get("prompt") or item.get("question") or item.get("input")
        if not prompt:
            continue
        expected = item.get("answer") or item.get("label") or item.get("expected")
        records.append({"prompt": prompt, "answer": expected})
    return records


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """Return the first index where needle occurs in haystack; -1 if not found."""
    if not needle:
        return -1
    n = len(needle)
    m = len(haystack)
    if n > m:
        return -1
    for i in range(m - n + 1):
        if list(haystack[i : i + n]) == list(needle):
            return i
    return -1


class TTBCController:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_think_budget: int = DEFAULT_THINK_BUDGET,
        max_answer_tokens: int = DEFAULT_ANSWER_MAX,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.seed = seed
        self.verbose = verbose

        # Safety cap
        self.max_think_budget = int(max_think_budget)
        self.max_answer_tokens = int(max_answer_tokens)

        print(f"Loading model: {model_name}")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.think_stop_str = "</think>"
        self.answer_stop_str = "</answer>"

        # Token-id patterns for robust stripping
        self.think_end_ids = self.tokenizer.encode(self.think_stop_str, add_special_tokens=False)
        self.wait_ids = self.tokenizer.encode(WAIT_TEXT, add_special_tokens=False)

    def _decode_ids(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _build_prompt(self, user_prompt: str) -> str:
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>"
        )

    def _sampling_for_thinking(self, max_tokens: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=max_tokens,
            min_tokens=0,
            stop=[self.think_stop_str],
            skip_special_tokens=False,
            temperature=self.temperature,
            top_p=1.0,
            seed=self.seed,
        )

    def _sampling_for_answer(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_answer_tokens,
            min_tokens=1,
            stop=[self.answer_stop_str],
            skip_special_tokens=False,
            temperature=self.temperature,
            top_p=1.0,
            seed=self.seed,
        )

    def _strip_think_end(self, segment_ids: Sequence[int]) -> Tuple[List[int], bool]:
        """
        Strip </think> (and everything after it) from a generated segment.
        Returns (stripped_ids, hit_delimiter).
        """
        idx = _find_subsequence(segment_ids, self.think_end_ids)
        if idx == -1:
            return list(segment_ids), False
        return list(segment_ids[:idx]), True

    def _generate_answer(self, think_context: str) -> Tuple[str, int]:
        """
        Close thinking, open <answer>, and generate.
        Strip any accidental </answer> if the backend returns it.
        """
        prompt = think_context + "</think>\n<answer>"
        
        if self.verbose:
            print("\n" + "="*80)
            print("ANSWER GENERATION PROMPT:")
            print("="*80)
            print(prompt)
            print("="*80)
        
        outputs = self.llm.generate([prompt], self._sampling_for_answer(), use_tqdm=False)
        out = outputs[0].outputs[0]
        txt = out.text
        
        if self.verbose:
            print("\n" + "="*80)
            print("ANSWER GENERATION RESPONSE:")
            print("="*80)
            print(txt)
            print("="*80 + "\n")
        
        if self.answer_stop_str in txt:
            txt = txt.split(self.answer_stop_str, 1)[0]
        return txt, len(out.token_ids)

    # -------------------------
    # TTBC1
    # -------------------------

    def _ttbc1_single(
        self,
        user_prompt: str,
        n_wait: int,
        max_think_tokens: Optional[int] = None,
    ) -> Tuple[SampleResult, Optional[DetailedResponse]]:
        """
        TTBC1: Wait & Think More
        
        Baseline (max_think_tokens=None, right of star):
          - When model generates </think>, suppress it and append "\nWait"
          - Repeat n_wait times, then stop thinking
        
        Budget control (max_think_tokens set, left of star):
          - If thinking < max_think_tokens and model wants to stop: append Wait and continue
          - If thinking >= max_think_tokens: truncate to exact budget and stop
        
        Returns: (SampleResult, DetailedResponse if verbose else None)
        """
        prefix = self._build_prompt(user_prompt)
        thought_ids: List[int] = []
        thinking_rounds = []

        if self.verbose:
            print("\n" + "="*80)
            print("INITIAL THINKING PROMPT:")
            print("="*80)
            print(prefix)
            print("="*80 + "\n")

        # Baseline: no budget forcing, only Wait intervention
        if max_think_tokens is None:
            waits_left = int(n_wait)
            round_num = 0
            
            for _ in range(n_wait + 1):
                context = prefix + self._decode_ids(thought_ids)
                outputs = self.llm.generate(
                    [context], 
                    self._sampling_for_thinking(self.max_think_budget), 
                    use_tqdm=False
                )
                out = outputs[0].outputs[0]
                
                seg_ids_raw = list(out.token_ids)
                seg_ids, hit_delim = self._strip_think_end(seg_ids_raw)
                
                round_info = {
                    "round": round_num,
                    "context_length": len(thought_ids),
                    "generated_tokens": len(seg_ids_raw),
                    "stripped_tokens": len(seg_ids),
                    "hit_delimiter": hit_delim,
                    "finish_reason": out.finish_reason,
                    "response_text": self._decode_ids(seg_ids_raw),
                }
                
                if self.verbose:
                    print(f"\n--- Thinking Round {round_num} ---")
                    print(f"Context length: {len(thought_ids)} tokens")
                    print(f"Generated: {len(seg_ids_raw)} tokens")
                    print(f"After stripping: {len(seg_ids)} tokens")
                    print(f"Hit delimiter: {hit_delim}")
                    print(f"Finish reason: {out.finish_reason}")
                    print("Response:")
                    print(self._decode_ids(seg_ids_raw))
                    print("-" * 40)
                
                thought_ids.extend(seg_ids)
                
                # If model wants to end thinking and we have Wait budget left
                if (hit_delim or out.finish_reason == "stop") and waits_left > 0:
                    thought_ids.extend(self.wait_ids)
                    round_info["wait_appended"] = len(self.wait_ids)
                    waits_left -= 1
                    
                    if self.verbose:
                        print(f"Appending WAIT ({len(self.wait_ids)} tokens): {WAIT_TEXT}")
                    
                    thinking_rounds.append(round_info)
                    round_num += 1
                    continue
                
                thinking_rounds.append(round_info)
                finish_reason = "stop"
                break
        
        # Budget control: force exact thinking tokens
        else:
            budget = int(max_think_tokens)
            round_num = 0
            
            while len(thought_ids) < budget:
                remaining = budget - len(thought_ids)
                context = prefix + self._decode_ids(thought_ids)
                outputs = self.llm.generate(
                    [context],
                    self._sampling_for_thinking(remaining),
                    use_tqdm=False
                )
                out = outputs[0].outputs[0]
                
                seg_ids_raw = list(out.token_ids)
                seg_ids, hit_delim = self._strip_think_end(seg_ids_raw)
                
                round_info = {
                    "round": round_num,
                    "context_length": len(thought_ids),
                    "budget_remaining": remaining,
                    "generated_tokens": len(seg_ids_raw),
                    "stripped_tokens": len(seg_ids),
                    "hit_delimiter": hit_delim,
                    "finish_reason": out.finish_reason,
                    "response_text": self._decode_ids(seg_ids_raw),
                }
                
                if self.verbose:
                    print(f"\n--- Thinking Round {round_num} (Budget Control) ---")
                    print(f"Context length: {len(thought_ids)} tokens")
                    print(f"Budget remaining: {remaining} tokens")
                    print(f"Generated: {len(seg_ids_raw)} tokens")
                    print(f"After stripping: {len(seg_ids)} tokens")
                    print(f"Hit delimiter: {hit_delim}")
                    print(f"Finish reason: {out.finish_reason}")
                    print("Response:")
                    print(self._decode_ids(seg_ids_raw))
                    print("-" * 40)
                
                # Truncate to remaining budget
                seg_ids = seg_ids[:remaining]
                thought_ids.extend(seg_ids)
                
                # If we reached budget, force stop
                if len(thought_ids) >= budget:
                    thinking_rounds.append(round_info)
                    finish_reason = "budget_forced"
                    break
                
                # If model ended early, inject Wait to continue
                if hit_delim or out.finish_reason == "stop":
                    remaining_after = budget - len(thought_ids)
                    wait_to_add = min(len(self.wait_ids), remaining_after)
                    thought_ids.extend(self.wait_ids[:wait_to_add])
                    round_info["wait_appended"] = wait_to_add
                    
                    if self.verbose:
                        print(f"Model ended early. Appending WAIT ({wait_to_add} tokens)")
                
                thinking_rounds.append(round_info)
                round_num += 1
            
            # Ensure exact budget (truncate if exceeded)
            thought_ids = thought_ids[:budget]
            finish_reason = "budget_forced"

        thought_text = self._decode_ids(thought_ids)
        answer_text, answer_tokens = self._generate_answer(prefix + thought_text)

        result = SampleResult(
            prompt=user_prompt,
            expected=None,
            answer=answer_text,
            extracted_answer=extract_answer(answer_text),
            thinking_tokens=len(thought_ids),
            answer_tokens=answer_tokens,
            thought=thought_text,
            n_wait=int(n_wait),
            max_think_tokens=max_think_tokens,
            finish_reason=finish_reason,
        )
        
        detailed = None
        if self.verbose:
            detailed = DetailedResponse(
                user_prompt=user_prompt,
                expected_answer=None,
                full_thinking_prompt=prefix,
                thinking_response=thought_text,
                full_answer_prompt=prefix + thought_text + "</think>\n<answer>",
                answer_response=answer_text,
                extracted_answer=extract_answer(answer_text),
                thinking_tokens=len(thought_ids),
                answer_tokens=answer_tokens,
                n_wait=int(n_wait),
                max_think_tokens=max_think_tokens,
                finish_reason=finish_reason,
                thinking_rounds=thinking_rounds,
            )
        
        return result, detailed

    # -------------------------
    # TTBC2
    # -------------------------

    def _ttbc2_single(self, user_prompt: str, t_exact: int) -> Tuple[SampleResult, Optional[DetailedResponse]]:
        """
        TTBC2: Exact Thinking Tokens
        
        Enforce exact thinking token budget of t_exact:
          - If thinking < t_exact and model wants to stop: append Wait and continue
          - If thinking >= t_exact: truncate to exact budget
        
        Returns: (SampleResult, DetailedResponse if verbose else None)
        """
        t_exact = int(t_exact)
        if t_exact <= 0:
            raise ValueError(f"t_exact must be > 0, got {t_exact}")
        if t_exact > self.max_think_budget:
            raise ValueError(
                f"t_exact ({t_exact}) > max_think_budget ({self.max_think_budget}). "
                "Increase --max-think-budget to run TTBC2 with this value."
            )

        prefix = self._build_prompt(user_prompt)
        thought_ids: List[int] = []
        thinking_rounds = []

        if self.verbose:
            print("\n" + "="*80)
            print("INITIAL THINKING PROMPT (TTBC2):")
            print("="*80)
            print(prefix)
            print("="*80 + "\n")

        round_num = 0
        while len(thought_ids) < t_exact:
            remaining = t_exact - len(thought_ids)
            context = prefix + self._decode_ids(thought_ids)
            outputs = self.llm.generate(
                [context],
                self._sampling_for_thinking(remaining),
                use_tqdm=False
            )
            out = outputs[0].outputs[0]

            seg_ids_raw = list(out.token_ids)
            seg_ids, hit_delim = self._strip_think_end(seg_ids_raw)

            round_info = {
                "round": round_num,
                "context_length": len(thought_ids),
                "t_exact_remaining": remaining,
                "generated_tokens": len(seg_ids_raw),
                "stripped_tokens": len(seg_ids),
                "hit_delimiter": hit_delim,
                "finish_reason": out.finish_reason,
                "response_text": self._decode_ids(seg_ids_raw),
            }

            if self.verbose:
                print(f"\n--- Thinking Round {round_num} (TTBC2) ---")
                print(f"Context length: {len(thought_ids)} tokens")
                print(f"Remaining to t_exact: {remaining} tokens")
                print(f"Generated: {len(seg_ids_raw)} tokens")
                print(f"After stripping: {len(seg_ids)} tokens")
                print(f"Hit delimiter: {hit_delim}")
                print(f"Finish reason: {out.finish_reason}")
                print("Response:")
                print(self._decode_ids(seg_ids_raw))
                print("-" * 40)

            # Truncate to remaining budget
            seg_ids = seg_ids[:remaining]
            thought_ids.extend(seg_ids)

            # If reached exact budget, stop
            if len(thought_ids) >= t_exact:
                thinking_rounds.append(round_info)
                break

            # If model ended thinking early, inject Wait and continue
            if hit_delim or out.finish_reason == "stop":
                remaining_after = t_exact - len(thought_ids)
                wait_to_add = min(len(self.wait_ids), remaining_after)
                thought_ids.extend(self.wait_ids[:wait_to_add])
                round_info["wait_appended"] = wait_to_add
                
                if self.verbose:
                    print(f"Model ended early. Appending WAIT ({wait_to_add} tokens): {WAIT_TEXT}")
            
            thinking_rounds.append(round_info)
            round_num += 1

        # Ensure exact guarantee (truncate if somehow exceeded)
        thought_ids = thought_ids[:t_exact]
        thought_text = self._decode_ids(thought_ids)

        answer_text, answer_tokens = self._generate_answer(prefix + thought_text)
        
        result = SampleResult(
            prompt=user_prompt,
            expected=None,
            answer=answer_text,
            extracted_answer=extract_answer(answer_text),
            thinking_tokens=len(thought_ids),
            answer_tokens=answer_tokens,
            thought=thought_text,
            t_exact=t_exact,
            finish_reason="exact",
        )
        
        detailed = None
        if self.verbose:
            detailed = DetailedResponse(
                user_prompt=user_prompt,
                expected_answer=None,
                full_thinking_prompt=prefix,
                thinking_response=thought_text,
                full_answer_prompt=prefix + thought_text + "</think>\n<answer>",
                answer_response=answer_text,
                extracted_answer=extract_answer(answer_text),
                thinking_tokens=len(thought_ids),
                answer_tokens=answer_tokens,
                t_exact=t_exact,
                finish_reason="exact",
                thinking_rounds=thinking_rounds,
            )
        
        return result, detailed

    # -------------------------
    # Batch runners
    # -------------------------

    def run_ttbc1(
        self,
        prompts: Sequence[Dict[str, Optional[str]]],
        n_wait_values: Sequence[int],
        max_think_tokens_values: Sequence[Optional[int]],
        save_response_path: Optional[str] = None,
    ) -> List[SampleResult]:
        results: List[SampleResult] = []
        detailed_responses: List[Dict[str, Any]] = []
        total = len(prompts) * len(n_wait_values) * len(max_think_tokens_values)
        print(f"Running TTBC1 on {len(prompts)} prompts. Total runs: {total}")

        done = 0
        for record in prompts:
            prompt = record.get("prompt") or ""
            expected = record.get("answer")
            for cap in max_think_tokens_values:
                for n_wait in n_wait_values:
                    res, detailed = self._ttbc1_single(prompt, n_wait=n_wait, max_think_tokens=cap)
                    res.expected = expected
                    if expected is not None:
                        res.accuracy = normalize(res.extracted_answer) == normalize(expected)
                    results.append(res)
                    
                    if save_response_path and detailed:
                        detailed.expected_answer = expected
                        detailed.accuracy = res.accuracy
                        detailed_responses.append(asdict(detailed))
                    
                    done += 1
                    if done % 10 == 0:
                        print(f"Processed {done}/{total}")
        
        if save_response_path and detailed_responses:
            self._save_detailed_responses(save_response_path, detailed_responses)
        
        return results

    def run_ttbc2(
        self,
        prompts: Sequence[Dict[str, Optional[str]]],
        t_exact_values: Sequence[int],
        save_response_path: Optional[str] = None,
    ) -> List[SampleResult]:
        results: List[SampleResult] = []
        detailed_responses: List[Dict[str, Any]] = []
        total = len(prompts) * len(t_exact_values)
        print(f"Running TTBC2 on {len(prompts)} prompts. Total runs: {total}")

        done = 0
        for record in prompts:
            prompt = record.get("prompt") or ""
            expected = record.get("answer")
            for t_exact in t_exact_values:
                res, detailed = self._ttbc2_single(prompt, t_exact=t_exact)
                res.expected = expected
                if expected is not None:
                    res.accuracy = normalize(res.extracted_answer) == normalize(expected)
                results.append(res)
                
                if save_response_path and detailed:
                    detailed.expected_answer = expected
                    detailed.accuracy = res.accuracy
                    detailed_responses.append(asdict(detailed))
                
                done += 1
                if done % 10 == 0:
                    print(f"Processed {done}/{total}")
        
        if save_response_path and detailed_responses:
            self._save_detailed_responses(save_response_path, detailed_responses)
        
        return results

    def _save_detailed_responses(self, path: str, detailed_responses: List[Dict[str, Any]]) -> None:
        """Save detailed responses to JSON file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(detailed_responses, indent=2, ensure_ascii=False))
        print(f"Saved {len(detailed_responses)} detailed responses to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TTBC experiments (TTBC1 / TTBC2)")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF repo or local path")
    common.add_argument("--tensor-parallel-size", type=int, default=1)
    common.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    common.add_argument("--temperature", type=float, default=0.0)
    common.add_argument("--seed", type=int, default=None)
    common.add_argument("--dataset", type=str, default=None)
    common.add_argument("--max-samples", type=int, default=None)
    common.add_argument("--output", type=str, default=None)
    common.add_argument("--max-think-budget", type=int, default=DEFAULT_THINK_BUDGET)
    common.add_argument("--max-answer-tokens", type=int, default=DEFAULT_ANSWER_MAX)
    common.add_argument("--verbose", action="store_true", help="Print prompts and model responses")
    common.add_argument("--save-response", type=str, default=None, 
                       help="Path to save detailed prompt/response information in JSON format")

    p1 = sub.add_parser("ttbc1", parents=[common], help="TTBC1: Wait & Think More")
    p1.add_argument("--n-wait", type=int, nargs="+", required=True,
                   help="Number of times to append Wait when model tries to end thinking")
    p1.add_argument(
        "--max-think-tokens",
        type=str,
        nargs="+",
        default=["none"],
        help="Thinking token budgets for forcing (use 'none' for baseline without budget control)",
    )

    p2 = sub.add_parser("ttbc2", parents=[common], help="TTBC2: Exact Thinking Tokens")
    p2.add_argument("--t-exact", type=int, nargs="+", required=True,
                   help="Exact thinking token budgets to enforce")

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if args.max_samples is not None:
        dataset = dataset[: args.max_samples]

    # Enable verbose mode if either --verbose or --save-response is specified
    verbose_mode = args.verbose or (args.save_response is not None)

    controller = TTBCController(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        seed=args.seed,
        max_think_budget=args.max_think_budget,
        max_answer_tokens=args.max_answer_tokens,
        verbose=verbose_mode,
    )

    if args.command == "ttbc1":
        caps: List[Optional[int]] = [None if v.lower() == "none" else int(v) for v in args.max_think_tokens]
        results = controller.run_ttbc1(dataset, args.n_wait, caps, save_response_path=args.save_response)
    else:
        results = controller.run_ttbc2(dataset, args.t_exact, save_response_path=args.save_response)

    output_records = [asdict(r) for r in results]
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output_records, indent=2, ensure_ascii=False))
        print(f"Saved {len(output_records)} records to {out_path}")
    else:
        print(json.dumps(output_records[:2], indent=2, ensure_ascii=False))
        if len(output_records) > 2:
            print(f"... ({len(output_records) - 2} more rows)")


if __name__ == "__main__":
    main()