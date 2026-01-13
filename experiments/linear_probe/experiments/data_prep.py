"""
Utilities to build word lists for linear probe experiments.

Outputs:
- words_perturbed.txt: confusable Cyrillic/Greek perturbations with a controllable ratio.
- words_random.txt: random a-z strings with the same length distribution as the base list.
- words_special.txt: random strings from the special symbol alphabet `_PGO#Xo+=.B*-@%&^`.
"""

from __future__ import annotations

import argparse
import random
import string
from pathlib import Path
from typing import Iterable

from .. import BASE_DIR
from ..normalizer import CONFUSABLE_TO_ASCII

SPECIAL_ALPHABET = "_PGO#Xo+=.B*-@%&^"


def load_words(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def save_words(path: Path, words: Iterable[str]) -> None:
    path.write_text("\n".join(words))


def _invert_confusable_map() -> dict[str, list[str]]:
    ascii_to_confusable: dict[str, list[str]] = {}
    for confusable, ascii_char in CONFUSABLE_TO_ASCII.items():
        ascii_to_confusable.setdefault(ascii_char, []).append(confusable)
    return ascii_to_confusable


def perturb_token(token: str, ratio: float, rng: random.Random, ascii_to_confusable: dict[str, list[str]]) -> str:
    chars: list[str] = []
    for ch in token:
        candidates = ascii_to_confusable.get(ch, ascii_to_confusable.get(ch.lower(), []))
        if candidates and rng.random() < ratio:
            chosen = rng.choice(candidates)
            # keep the original casing if possible
            if ch.isupper() and chosen.isalpha() and chosen.islower():
                chosen = chosen.upper()
            chars.append(chosen)
        else:
            chars.append(ch)
    return "".join(chars)


def perturb_words(words: list[str], ratio: float, rng: random.Random) -> list[str]:
    ascii_to_confusable = _invert_confusable_map()
    return [perturb_token(word, ratio, rng, ascii_to_confusable) for word in words]


def random_words(words: list[str], rng: random.Random) -> list[str]:
    alphabet = string.ascii_lowercase
    return [
        "".join(rng.choice(alphabet) for _ in range(len(word)))
        for word in words
    ]


def _weighted_special_choice(rng: random.Random) -> str:
    # # Bias towards "_", "#", "B" as requested.
    # weights = [3 if ch in {"_", "#", "B"} else 1 for ch in SPECIAL_ALPHABET]
    weights = [1 if ch in {"_", "#", "B"} else 1 for ch in SPECIAL_ALPHABET]
    population = list(SPECIAL_ALPHABET)
    total = sum(weights)
    pick = rng.uniform(0, total)
    cumulative = 0.0
    for ch, w in zip(population, weights):
        cumulative += w
        if pick <= cumulative:
            return ch
    return population[-1]


def special_words(words: list[str], rng: random.Random) -> list[str]:
    return [
        "".join(_weighted_special_choice(rng) for _ in range(len(word)))
        for word in words
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate word lists for linear probe experiments.")
    parser.add_argument(
        "--base_path",
        type=Path,
        default=BASE_DIR / "words.txt",
        help="Path to the clean English word list.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=BASE_DIR,
        help="Directory to save generated word lists.",
    )
    parser.add_argument(
        "--perturb_ratio",
        type=float,
        default=0.9,
        help="Probability of perturbing each ASCII character with a confusable counterpart.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20250315,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    words = load_words(args.base_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    perturbed = perturb_words(words, args.perturb_ratio, rng)
    rand_words = random_words(words, rng)
    special = special_words(words, rng)

    save_words(args.output_dir / "words_perturbed.txt", perturbed)
    save_words(args.output_dir / "words_random.txt", rand_words)
    save_words(args.output_dir / "words_special.txt", special)

    print(f"Base words: {len(words)}")
    print(f"Saved perturbed -> {args.output_dir / 'words_perturbed.txt'}")
    print(f"Saved random    -> {args.output_dir / 'words_random.txt'}")
    print(f"Saved special   -> {args.output_dir / 'words_special.txt'}")


if __name__ == "__main__":
    main()
