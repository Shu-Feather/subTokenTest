# SubTokenTest Datasets

This directory contains JSONL files produced by each task’s `create_datasets.py` helper. Every line holds a `question` (the exact prompt fed to models, already containing `<answer>` tags) and an `answer` (ground truth without tags).

## Task Overviews & Examples

- **Adversarial Prompt Canonicalization** (`adversarial_prompt_datasets.jsonl`): Given a perturbed/jailbreak-style text, the model must canonicalize it back to the original intent inside `<answer>…</answer>`.  
  *Example*: Perturbed: “1gn0r3-pr3v10us-1nstruct10ns” → `<answer>ignore previous instructions</answer>`.

- **Biological Sequence Tasks** (`bio_seq_datasets.jsonl`): Four task types:
  - `dna_complement`: DNA complement (A↔T, C↔G).  
    *Example*: Input: `ATCG` → `<answer>TAGC</answer>`.
  - `rna_complement`: RNA complement (A↔U, C↔G).  
    *Example*: Input: `AUCG` → `<answer>UAGC</answer>`.
  - `protein_three_to_one`: Three-letter AA codes to one-letter.  
    *Example*: `GLY-ARG-PHE` → `<answer>GRF</answer>`.
  - `protein_one_to_three`: One-letter AA codes to three-letter.  
    *Example*: `GRF` → `<answer>GLY-ARG-PHE</answer>`.

- **Cipher & Decipher** (`cipher_decipher_datasets.jsonl`): Four task types:
  - `morse_encode`: Plain text → Morse. *Example*: “HELLO” → `<answer>.... . .-.. .-.. ---</answer>`.
  - `morse_decode`: Morse → Plain text. *Example*: “.... . .-.. .-.. ---” → `<answer>HELLO</answer>`.
  - `caesar_encode`: Caesar shift encode. *Example*: Text “abc”, shift 3 → `<answer>def</answer>`.
  - `caesar_decode`: Caesar shift decode. *Example*: Cipher “def”, shift 3 → `<answer>abc</answer>`.

- **Context-Aware Redaction** (`context_aware_redaction_datasets.jsonl`): Redact sensitive items (phone, ID card, credit card) using masking rules while preserving other text.  
  *Example*: Phone “+1 38024553160” → `<answer>+1 380****3160</answer>` inside the redacted context.

- **Gomoku Board Analysis** (`gomoku_linear_datasets.jsonl`, `gomoku_diagonal_datasets.jsonl`): Analyze a board (B/W/E) and output exactly one of `WHITE_WINS`, `BLACK_WINS`, or `NO_WINNER` in `<answer>…</answer>`.  
  *Example*: If black has five in a row → `<answer>BLACK_WINS</answer>`.

- **Map Navigation QA** (`map_nav_sokoban_datasets.jsonl`, `map_nav_frozenlake_datasets.jsonl`): Given a Sokoban or FrozenLake map plus a question, answer inside `<answer>…</answer>`.  
  *Example*: “How many boxes remain to be placed?” → `<answer>2</answer>`.

- **RSA Randomart Difference Detection** (`randomart_{3,5,7,9}_diff_datasets.jsonl`): Compare two RSA fingerprint patterns and list all differences as coordinates and replacements.  
  *Example*: `<answer>(7, 3):   -> o</answer>`.

- **Aligned Table Formatting** (`table_datasets.jsonl`): Produce a correctly aligned table (latex/markdown/text) from row data, wrapped in `<answer>…</answer>`.  
  *Example*: Markdown table for given rows.

- **Binary Tree Reasoning** (`tree_structure_datasets.jsonl`, `tree_path_datasets.jsonl`):
  - Structure queries (parent/left/right/num_nodes) → `<answer>…</answer>`.  
    *Example*: “What is the parent of node 4?” → `<answer>2</answer>`.
  - Path finding between two nodes → `<answer>4 -> 2 -> 1 -> 3 -> 6</answer>`.

- **Typewriter Tasks** (`typewriter_datasets.jsonl`): Two task types:
  - Encoding task (progressive typing): show stepwise reveal. *Example*: “cat” → `<answer>c→ca→cat</answer>`.
  - Decoding task (backspace handling): process logs with “←”. *Example*: `h e l l o ← ← k o` → `<answer>helko</answer>`.

## Dataset Summary

| File | Samples | Notes |
| --- | --- | --- |
| adversarial_prompt_datasets.jsonl | 100 | Perturbed text → canonical form |
| bio_seq_datasets.jsonl | 100 | DNA/RNA/protein conversions (4 task types) |
| cipher_decipher_datasets.jsonl | 200 | Morse/Caesar encode/decode (4 task types) |
| context_aware_redaction_datasets.jsonl | 100 | Redact PII with masking rules |
| gomoku_diagonal_datasets.jsonl | 100 | Gomoku board state (diagonal-heavy set) |
| gomoku_linear_datasets.jsonl | 100 | Gomoku board state (linear-heavy set) |
| map_nav_frozenlake_datasets.jsonl | 100 | FrozenLake map QA |
| map_nav_sokoban_datasets.jsonl | 100 | Sokoban map QA |
| randomart_3_diff_datasets.jsonl | 100 | RSA randomart, 3 differences |
| randomart_5_diff_datasets.jsonl | 100 | RSA randomart, 5 differences |
| randomart_7_diff_datasets.jsonl | 100 | RSA randomart, 7 differences |
| randomart_9_diff_datasets.jsonl | 100 | RSA randomart, 9 differences |
| table_datasets.jsonl | 100 | Table alignment (latex/markdown/text) |
| tree_structure_datasets.jsonl | 100 | Tree structure queries |
| tree_path_datasets.jsonl | 100 | Tree path finding |
| typewriter_datasets.jsonl | 200 | Encoding / Decoding tasks |

