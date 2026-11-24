# factual_v2: factual substitution (rule-based), full scale

Each poisoned passage is a copy of a gold passage in which the number the question asks for is replaced by a
different value. Three identical copies per question. No language model is involved. This is version 2 of the
attack; it replaces `factual_v1`, whose number swap ignored context (see "Changes from v1").

## Files

| File | Rows | In Git | Contents |
|---|---|---|---|
| `poisoned_passages.jsonl` | 3,276 | yes | `passage_id`, `title` (the gold passage's title), `text` |
| `provenance.jsonl` | 3,276 | yes | one row per passage: question ID and type, source passage title, true value, false value, copy index, occurrences replaced, day-of-month flag, whether the value differs from v1's draw, number of non-gold paragraphs containing the same number |
| `rejected.jsonl` | 76 | yes | questions skipped by the v2 rules, with the reason |
| `run_manifest.json` | 1 | yes | settings and counts |
| `v1_vs_v2.md` | — | yes | per-check comparison with factual_v1 and examples |

## How it was made

- Script: `src/data/attacks/factual_substitution_v2.py`, seed 42, 3 copies per question, input: the 7,405 HotpotQA
  distractor dev questions (dev and test splits).
- The first number in the gold answer is read (without a trailing comma). The first gold passage containing it is
  the target. Every standalone occurrence is replaced (also before a comma or full stop; never digits of a larger
  number such as "1,000" or of a decimal such as "1.5"); a following ordinal suffix is rewritten.
- New value: years 1000–2030 move by 5–20 years and never past 2026; decades move by whole decades; a day of the
  month stays 1–28; other numbers are scaled by 0.55, 0.7, 1.35, 1.6 or 2.0. The value may not already appear in
  the passage. For questions v1 handled correctly, v2 draws the same value as v1 (same random sequence); 167 values
  were redrawn because v1's value was invalid.
- Skipped: answer number part of a decimal (29), number one end of a range or hyphenated code in the passage (47),
  no number in the answer (6,235), number not found in a gold passage (2).
- Every passage is checked before it is written: false value present, true value absent, no impossible date.
- Command: `python src/data/attacks/factual_substitution_v2.py` (CPU, seconds).

## Counts

| | dev | test | all |
|---|---|---|---|
| Poisoned questions | 896 | 196 | 1,092 |
| Poisoned passages | 2,688 | 588 | 3,276 |

Question types: bridge 1,054, comparison 38. Occurrences replaced per passage: 1 in 966 questions, 2 in 94, 3+ in 32.

## Quality checks (2026-11-23)

- Structure: 3,276 passages = 3,276 provenance rows; passage IDs unique; 3 identical copies per question; every
  passage keeps its source title; one false value per question.
- Content, own rule: false value present in 1,092 / 1,092 questions, true value present in 0.
- Content, the LLM pipeline's whole-token match: false value found in 999 / 1,092 (91.5%; the rest are joined to
  letters or a suffix, e.g. "A41" → "A29", "12th" → "19th"); true value found in 1 ("3.11", a nickname of 11 March).
- False value of the same kind as the true one (4-digit / other number): 98.1%.
- Reproducibility: rerunning `factual_substitution.py` (v1) on the same input reproduces the released `factual_v1`
  byte for byte, so the v1/v2 comparison is exact. 16 hand-built unit tests cover each v1 defect.
- Manual read, 30 random questions: 26 clean and plausible; 4 plausibility issues (a 3-digit year scaled as a count,
  "934 to 961" → "514 to 961"; two dates that clash with history; one unrelated equal number changed).

## Changes from v1

| Check (per question) | v1 | v2 |
|---|---|---|
| Questions | 1,099 | 1,092 |
| True number still in the passage | 40 | 0 |
| Impossible date added ("September 48") | 47 | 0 |
| Comma lost from the passage | 123 | 2 (both thousands values that became smaller than 1,000, e.g. 1,462 → 804; correct) |
| Year after 2026 | 9 | 0 |
| False value already in the source passage | 20 | 0 |
| One end of a range changed | 46 | 0 |
| Decade broken ("1970s" → "1975s") | 6 | 0 |
| Year before 1500 scaled as a count | 10 | 0 |

1,021 questions are in both versions (924 with the same false value, 793 with identical text); 78 v1 questions are
dropped (decimal 29, range 47, not found 2) and 71 are new (v1 missed them because of the trailing comma).

## Does the poison reach the retriever?

`smoke_test.py` over all 1,092 questions (output `results/smoke_test/full_factual_v2/`, not in Git):

| Retriever | poison in top 5 | poison in top 10 | poison above its target | gold pushed out of top 5 |
|---|---|---|---|---|
| Dense (e5-base-v2, cosine) | 0.59 | 0.70 | 0.39 | 0.01 |
| BM25 (Lucene, k1 1.5, b 0.75) | 0.42 | 0.53 | 0.03 | 0.00 |

Same as v1 (dense 0.59 / 0.70, BM25 0.42 / 0.53). Dev and test agree (dense poison@5 0.58 / 0.61).

## Known limitations

- Plausibility is not checked automatically (see the manual read).
- An unrelated number equal to the answer number in the same passage is changed too (148 answers are a single digit).
- 377 questions have a non-gold paragraph containing the same number (a weak residual measure: numbers are common).
- Only answers that contain a number are attacked (14.7% of questions).
