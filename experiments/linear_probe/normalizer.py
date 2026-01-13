CONFUSABLE_TO_ASCII = {
    # Cyrillic -> ASCII
    "а": "a", "А": "A",
    "е": "e", "Е": "E",
    "о": "o", "О": "O",
    "р": "p", "Р": "P",
    "с": "c", "С": "C",
    "у": "y", "У": "Y",
    "х": "x", "Х": "X",
    "і": "i", "І": "I",
    "ј": "j", "Ј": "J",
    "ѕ": "s", "Ѕ": "S",
    "г": "r", "Г": "R",
    "ь": "b", "Ь": "B",
    "ъ": "b", "Ъ": "B",
    "ӏ": "l",
    "ԁ": "d", "Ԃ": "D",
    "ԛ": "q", "Ԝ": "W", "ԝ": "w",
    # Greek -> ASCII
    "ν": "v", "Ν": "N",
    "υ": "u", "Υ": "Y",
    "ο": "o", "Ο": "O",
    "ρ": "p", "Ρ": "P",
    "χ": "x", "Χ": "X",
    "κ": "k", "Κ": "K",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Μ": "M", "Τ": "T",
    "α": "a", "β": "b", "γ": "y", "δ": "d", "ε": "e", "ζ": "z", "η": "n", "ι": "i",
    "λ": "l", "μ": "m", "ς": "s", "τ": "t", "φ": "f", "ω": "w",
}


def normalize_confusables(token: str) -> str:
    """
    Replace confusable Cyrillic/Greek characters with their ASCII look-alikes.
    """
    return "".join(CONFUSABLE_TO_ASCII.get(ch, ch) for ch in token)


def build_char_stats(tokens: list[str]) -> tuple[dict[str, int], int, int]:
    """
    Build char2label, max_num_each_char, max_length_each_token from normalized tokens.
    """
    char2label: dict[str, int] = {}
    max_num_each_char = 0
    max_length_each_token = 0
    for token in tokens:
        max_length_each_token = max(max_length_each_token, len(token))
        for char in token:
            if char not in char2label:
                char2label[char] = len(char2label)
        if token:
            counts = [token.count(c) for c in set(token)]
            max_num_each_char = max(max_num_each_char, max(counts))
    return char2label, max_num_each_char, max_length_each_token
