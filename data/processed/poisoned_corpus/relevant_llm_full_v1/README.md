# relevant_llm_full_v1 — full-scale highly relevant poisoning (HotpotQA, dev + test)

Knowledge-base poisoning set for the project "Can We Trust the Verifier? Verification Independence in RAG Under
Knowledge-Base Poisoning". For each poisoned question, a new passage is written that answers the question with a
false value, built around the names and terms in the question and supporting facts from the question's gold
passages. It is filed under the title of the first gold passage that states the answer. It is built to be as
relevant to the question as possible, so a retriever ranks it high. Generated with the final prompt (`b0a1af93cf`)
and checks developed on the pilot set `relevant_llm_v1` (revision 4 of the generator, `--derived_forms`), which
itself mixes earlier prompt versions and models; this set uses one prompt version and one model throughout.

## What is in this folder

| File | Rows | In Git | What it is |
|---|---|---|---|
| `poisoned_passages.jsonl` | 9,849 | yes | the fake passages: `passage_id`, `title`, `text` |
| `provenance.jsonl` | 9,849 | yes | one row per fake passage: question ID and type, target passage, true value, false value, copy index, model, prompt version |
| `run_manifest.json` | — | yes | settings and summary counts |
| `README.md` | — | yes | this data card |
| `rejected.jsonl` | 1,153 | no (Drive) | eligible questions whose output failed the checks, with reasons |
| `skipped.jsonl` | 2,969 | no (Drive) | questions not eligible for this attack, with the reason |
| `checkpoint.jsonl` | 4,437 lines (4,436 questions) | no (Drive) | raw record per attempted question (source of truth; every other file is rebuilt from it) |
| `inspect.md` | — | no (Drive) | human-readable sheet: gold passage and all versions for every question |
| `corpus.jsonl`, `corpus_manifest.json` | 76,430 | no (git-ignored) | clean corpus + fake passages; rebuild with `src/data/build_poisoned_corpus.py` |

Only the four "yes" files are in Git: the two data files that the corpus, retrieval, RAG and verifier scripts
read, plus the run manifest and this card. The "Drive" files are kept, with md5-checked backups, in the project's
Google Drive folder and are available on request. The search indexes (`data/indexes/relevant_llm_full_v1`) and the
retrieval check (`results/smoke_test/full_paraphrase_relevant`) are also kept on Drive; both can be rebuilt (see
below).

## How it was made

- Questions: HotpotQA distractor dev set, 7,405 questions, split with seed 42 into dev 5,924 / test 1,481
  (`data/processed/splits`).
- Script: `src/data/attacks/relevant_llm_r4.py --derived_forms` (built on `relevant_llm.py` and
  `contradiction_llm.py`), prompt version `b0a1af93cf`. One fake-passage target per question (the first gold
  passage stating the answer).
- Model: `openai/gpt-oss-120b` served with vLLM 0.34.0 on one NVIDIA A100-SXM4 80 GB (Colab Pro), MXFP4 weights
  (Marlin backend), 32 parallel requests.
- Settings: seed 42, `--split all`, `max_attempts` 3, 3 versions per question, maximum pairwise similarity between
  versions 0.8, and the limits recorded in `run_manifest.json` (`max_reference_similarity` 0.75,
  `max_reference_run` 12, `max_copied_run` 7, `min_question_coverage` 0.5).
- Run: 2026-11-21 02:30–04:05 UTC (about 47 questions per minute), after a 10-question smoke test (7 accepted).

Commands (the vLLM flags are the defaults of `scripts/run_poison_h100.sh` for the 120b model):
```
ATTACKS=relevant SPLIT=all WORKERS=32 bash scripts/run_poison_h100.sh attacks
#   = vllm serve openai/gpt-oss-120b --port 8000 --gpu-memory-utilization 0.95 --max-num-batched-tokens 1024 \
#       --max-model-len 16384 --max-num-seqs 64
#     python src/data/attacks/relevant_llm_r4.py --base_url http://127.0.0.1:8000/v1 --model openai/gpt-oss-120b \
#       --max_attempts 3 --split all --limit 100000 --workers 32 --derived_forms \
#       --out_dir data/processed/poisoned_corpus/relevant_llm_full_v1
python src/data/build_poisoned_corpus.py --poison_path data/processed/poisoned_corpus/relevant_llm_full_v1/poisoned_passages.jsonl \
  --out_dir data/processed/poisoned_corpus/relevant_llm_full_v1 --variant_name relevant_llm_full_v1
python src/retrieval/build_indexes.py --config configs/index_config_full.yaml --variants relevant_llm_full_v1
python src/retrieval/smoke_test.py --config configs/index_config_full.yaml --variants paraphrase_llm_full_v1 relevant_llm_full_v1 --tag full_paraphrase_relevant
```

### One question rejected at export (code fix on this branch)

`--derived_forms` accepts a list answer when every item appears, so question `5add352b5542990dbb2f7dc4` (false
answer "Peter Lawford, Joey Bishop, Bing Crosby", written "Peter Lawford, Joey Bishop, and Bing Crosby") was
accepted during generation. The shared export then re-checks the exact phrase and stopped with `AssertionError:
false answer missing`. During the run this was resolved by appending one corrected record to the append-only
checkpoint (status rejected, reason `export_check:false_answer_not_verbatim`); the original line is kept.
`relevant_llm_r4.py` now applies the same rule itself at export (with `--derived_forms`, an accepted question
whose versions lack the exact phrase is rejected instead of stopping the export). Re-exporting the original
checkpoint (without the added record) with the fixed script reproduces every output file of this folder byte for
byte, with no model calls.

## Counts

| | Dev | Test | Total |
|---|---|---|---|
| Questions | 5,924 | 1,481 | 7,405 |
| Not eligible | 2,377 | 592 | 2,969 |
| Eligible (attempted) | 3,547 | 889 | 4,436 |
| **Accepted (poisoned)** | **2,635 (74.3%)** | **648 (72.9%)** | **3,283 (74.0%)** |
| Rejected by the checks | 912 | 241 | 1,153 |
| Fake passages | 7,905 | 1,944 | 9,849 |

Not eligible (same rules for every LLM attack): answer appears in the gold passage title 2,492; yes/no answer 458;
answer too short 14; answer absent from the gold passages 5.

Accepted: bridge 3,152, comparison 131; one target per question (3,283 × 3 versions = 9,849). Attempts needed: 1
for 1,759 questions, 2 for 1,055, 3 for 469. For 563 accepted questions (17.1%) one of the question's own
distractor paragraphs still states the true answer.

Rejection reasons (a question can have several): copied a run of words from a reference 427; a name added 371;
copied a run of words from the question 313; true answer still present 309; a number word added 236; stand-in
sentence 200; a number added 192; replacement already in source 156; title bracket in text 107; false answer
absent 96; a variant of the true answer present 62; part of the true answer present 49; question mark introduced
39; answer only tacked on 36; too little of the question covered 25; 16 other reasons (each under 25) 184,
including the one export check above.

## Quality checks (2026-11-21)

- Structure: 9,849 passages = 9,849 provenance rows; all passage ids unique and none shared with the clean corpus;
  exactly 3 versions per question; every fake keeps its target title; one false value per question; one model and
  prompt version throughout; accepted ids in provenance = accepted ids in the checkpoint.
- Content (the pipeline's whole-token match on normalized text, enforced at export): the false value is stated in
  9,849 / 9,849 fake passages (100%); the true value remains in 0 / 9,849.
- False answer of the same kind as the true one (4-digit number / other number / text): 98.0%. No false answer
  contains the true one or the reverse.
- These are new passages, not rewrites: text similarity to the gold passage (difflib ratio) median 0.12 (90th
  percentile 0.29). Pairwise similarity between the 3 versions: mean 0.40, maximum 0.80.
- Manual audit, 50 random accepted questions (seed 2026, dev and test): 42 clean and plausible; 4 with awkward or
  meta wording that echoes the question (e.g. "The question mentions…", "the city named above"); 3 internally
  inconsistent (e.g. a Great Neck, New York school district placed "in Toronto, Canada"); 1 thin passage of filler
  sentences. None states the true answer.
- Agreement with the pilot: 24 of the pilot's 33 accepted questions are accepted again (same false value in 7;
  generation uses temperature 0.7).

## Does the poison reach the retriever?

Retrieval check over all 3,283 poisoned questions (`smoke_test.py`, e5-base-v2 dense and BM25 indexes built with
`configs/index_config_full.yaml`):

| Retriever | fake in top 5 | fake in top 10 | fakes in top 5 (mean) | gold in top 5: clean → poisoned | gold pushed out of top 5 | fake above its target | median rank of first fake | verdict |
|---|---|---|---|---|---|---|---|---|
| Dense (e5-base-v2, cosine) | 0.99 | 1.00 | 2.81 | 0.97 → 0.94 | 0.04 | 0.97 | 1 | surfaces |
| BM25 (Lucene, k1 1.5, b 0.75) | 0.99 | 1.00 | 2.86 | 0.94 → 0.89 | 0.06 | 0.98 | 1 | surfaces |

Dev and test give similar figures. "Fake above its target": a fake ranks higher than the gold passage whose title
it carries, or that passage is not in the top 20. This is by far the strongest of the four attacks: the first fake
passage has median rank 1, on average 2.8 of its 3 versions are in the top 5, and the gold passage is pushed out
of the top 5 for 4–6% of questions. Full figures: `results/smoke_test/full_paraphrase_relevant/summary.json`.

## Known limitations

- In the manual read, 4 of 50 questions had awkward or meta wording that echoes the question ("The question
  mentions…"), a trace a detector could pick up.
- The automatic checks do not test plausibility or internal consistency (see the audit above).
- 563 accepted questions (17.1%) keep a distractor paragraph that states the true answer.
- 1.9% of fake passages contain special characters that the gold passage does not.
- Comparison questions are few (131), because most comparison answers are yes/no or a passage title.

## Rebuilding the corpus and indexes

`load_hotpotqa.py` → `build_clean_corpus.py` → `build_poisoned_corpus.py` → `build_indexes.py` → `smoke_test.py`,
with the commands above. No GPU is needed for the corpus; the dense index needs one.
