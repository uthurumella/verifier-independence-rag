# paraphrase_llm_full_v1 — full-scale paraphrase poisoning (HotpotQA, dev + test)

Knowledge-base poisoning set for the project "Can We Trust the Verifier? Verification Independence in RAG Under
Knowledge-Base Poisoning". For each poisoned question, each gold passage that states the answer is rewritten in
new words (new sentence order and phrasing, same facts) with the answer replaced by a false value. Unlike the
contradiction attack, the fake passage is not a near-copy of the source: its word-sequence similarity to the
source may not exceed 0.75. Built with the same prompt, model, checks and seed as the pilot set
`paraphrase_llm_v1`; only the serving backend (a local vLLM server instead of the Groq API) and the scale changed.

## What is in this folder

| File | Rows | In Git | What it is |
|---|---|---|---|
| `poisoned_passages.jsonl` | 11,082 | yes | the fake passages: `passage_id`, `title`, `text` |
| `provenance.jsonl` | 11,082 | yes | one row per fake passage: question ID and type, source passage, true value, false value, copy index, model, prompt version |
| `run_manifest.json` | — | yes | settings and summary counts |
| `README.md` | — | yes | this data card |
| `rejected.jsonl` | 964 | no (Drive) | eligible questions whose output failed the checks, with reasons |
| `skipped.jsonl` | 2,969 | no (Drive) | questions not eligible for this attack, with the reason |
| `checkpoint.jsonl` | 4,436 | no (Drive) | raw record per attempted question (source of truth; every other file is rebuilt from it) |
| `inspect.md` | — | no (Drive) | human-readable sheet: source passage and all versions for every question |
| `corpus.jsonl`, `corpus_manifest.json` | 77,663 | no (git-ignored) | clean corpus + fake passages; rebuild with `src/data/build_poisoned_corpus.py` |

Only the four "yes" files are in Git: the two data files that the corpus, retrieval, RAG and verifier scripts
read, plus the run manifest and this card. The "Drive" files are kept, with md5-checked backups, in the project's
Google Drive folder and are available on request. The search indexes (`data/indexes/paraphrase_llm_full_v1`) and
the retrieval check (`results/smoke_test/full_paraphrase_relevant`) are also kept on Drive; both can be rebuilt
(see below).

## How it was made

- Questions: HotpotQA distractor dev set, 7,405 questions, split with seed 42 into dev 5,924 / test 1,481
  (`data/processed/splits`).
- Script: `src/data/attacks/paraphrase_llm.py` (built on `contradiction_llm.py`), prompt version `8547a126ef`.
- Model: `openai/gpt-oss-120b` served with vLLM 0.34.0 on one NVIDIA A100-SXM4 80 GB (Colab Pro), MXFP4 weights
  (Marlin backend), 32 parallel requests.
- Settings: seed 42, `--split all`, `max_attempts` 3, 3 versions per target passage, maximum pairwise similarity
  between versions 0.8, maximum similarity to the source passage 0.75.
- Run: 2026-11-21 00:49–02:29 UTC (about 44 questions per minute), after a 10-question smoke test (8 accepted).

Commands (the vLLM flags are the defaults of `scripts/run_poison_h100.sh` for the 120b model):
```
ATTACKS=paraphrase SPLIT=all WORKERS=32 bash scripts/run_poison_h100.sh attacks
#   = vllm serve openai/gpt-oss-120b --port 8000 --gpu-memory-utilization 0.95 --max-num-batched-tokens 1024 \
#       --max-model-len 16384 --max-num-seqs 64
#     python src/data/attacks/paraphrase_llm.py --base_url http://127.0.0.1:8000/v1 --model openai/gpt-oss-120b \
#       --max_attempts 3 --split all --limit 100000 --workers 32 --out_dir data/processed/poisoned_corpus/paraphrase_llm_full_v1
python src/data/build_poisoned_corpus.py --poison_path data/processed/poisoned_corpus/paraphrase_llm_full_v1/poisoned_passages.jsonl \
  --out_dir data/processed/poisoned_corpus/paraphrase_llm_full_v1 --variant_name paraphrase_llm_full_v1
python src/retrieval/build_indexes.py --config configs/index_config_full.yaml --variants paraphrase_llm_full_v1
python src/retrieval/smoke_test.py --config configs/index_config_full.yaml --variants paraphrase_llm_full_v1 relevant_llm_full_v1 --tag full_paraphrase_relevant
```

## Counts

| | Dev | Test | Total |
|---|---|---|---|
| Questions | 5,924 | 1,481 | 7,405 |
| Not eligible | 2,377 | 592 | 2,969 |
| Eligible (attempted) | 3,547 | 889 | 4,436 |
| **Accepted (poisoned)** | **2,749 (77.5%)** | **723 (81.3%)** | **3,472 (78.3%)** |
| Rejected by the checks | 798 | 166 | 964 |
| Fake passages | 8,754 | 2,328 | 11,082 |

Not eligible (same rules for every LLM attack): answer appears in the gold passage title 2,492; yes/no answer 458;
answer too short 14; answer absent from the gold passages 5.

Accepted: bridge 3,294, comparison 178; one target passage 3,250, two target passages 222 (3,694 target passages ×
3 versions = 11,082). Attempts needed: 1 for 2,046 questions, 2 for 986, 3 for 440. For 592 accepted questions
(17.1%) one of the question's own distractor paragraphs still states the true answer (not rewritten).

Rejection reasons (a question can have several): a required name missing 1,238; a required number missing 554;
length ratio 406; true answer still present 173; too close to the source 171; number added 144; part of the true
answer present 100; false answer absent 84; subject dropped 77; replacement already in source 26; versions too
similar 12; replacement overlaps answer 8; header or markup 6; fewer than 3 versions returned 2.

## Quality checks (2026-11-21)

- Structure: 11,082 passages = 11,082 provenance rows; all passage ids unique and none shared with the clean
  corpus; exactly 3 versions per target passage; every fake keeps its source title; one false value per question;
  one model and prompt version throughout; accepted ids in provenance = accepted ids in the checkpoint.
- Content (the pipeline's whole-token match on normalized text, enforced at export): the false value is stated in
  11,082 / 11,082 fake passages (100%); the true value remains in 0 / 11,082.
- False answer of the same kind as the true one (4-digit number / other number / text): 98.9%. No false answer
  contains the true one or the reverse.
- Text similarity to the source passage (difflib character ratio, a different measure from the 0.75 acceptance
  limit): median 0.28 (10th percentile 0.10, 90th 0.56, max 0.83); length relative to the source (words, as the
  length check counts them): median 0.93× (range 0.60–1.50×; the check allows 0.6–1.5×). Pairwise similarity
  between the 3 versions: mean 0.45, maximum 0.79.
- Manual audit, 50 random accepted questions (seed 2026, dev and test): 37 clean and plausible; 2 weak
  near-synonym swaps (e.g. "rocket" → "missile"); 3 implausible but self-consistent (e.g. the UK listed-building
  register named as the "National Register of Historic Places"); 7 internally inconsistent (e.g. the University of
  St Andrews placed "in Aberdeen"; a Bond-film album described as "Ethan Hunt movies"); 1 text defect (part of a
  Cyrillic name dropped). None states the true answer.
- Agreement with the pilot: 25 of the pilot's 30 accepted questions are accepted again (same false value in 10;
  generation uses temperature 0.7).

## Does the poison reach the retriever?

Retrieval check over all 3,472 poisoned questions (`smoke_test.py`, e5-base-v2 dense and BM25 indexes built with
`configs/index_config_full.yaml`):

| Retriever | fake in top 5 | fake in top 10 | gold in top 5: clean → poisoned | gold pushed out of top 5 | fake above its source | verdict |
|---|---|---|---|---|---|---|
| Dense (e5-base-v2, cosine) | 0.50 | 0.61 | 0.97 → 0.97 | 0.00 | 0.18 | surfaces |
| BM25 (Lucene, k1 1.5, b 0.75) | 0.42 | 0.53 | 0.95 → 0.94 | 0.01 | 0.22 | surfaces |

Dev and test give similar figures (dense poison@5 0.50 / 0.50; BM25 0.42 / 0.45). "Fake above its source": a fake
ranks higher than the passage it was made from, or that passage is not in the top 20. Rewording costs the fake
passages rank: they reach the top 5 less often than the contradiction set (dense 0.60 vs 0.50) and outrank the
passage they were made from less often (dense 0.18 vs 0.44). Full figures:
`results/smoke_test/full_paraphrase_relevant/summary.json`.

## Known limitations

- The automatic checks do not test plausibility or internal consistency; in the audit, 7 of 50 questions (14%) had
  an internal inconsistency and 3 more were implausible (see above).
- 592 accepted questions (17.1%) keep a distractor paragraph that states the true answer, which may weaken the
  attack for them.
- Typography: 2.8% of fake passages contain special characters that the source passage does not (output
  normalization turns special hyphens, curly quotes and odd spaces into plain ones unless the source uses them).
- Comparison questions are few (178), because most comparison answers are yes/no or a passage title.

## Rebuilding the corpus and indexes

`load_hotpotqa.py` → `build_clean_corpus.py` → `build_poisoned_corpus.py` → `build_indexes.py` → `smoke_test.py`,
with the commands above. No GPU is needed for the corpus; the dense index needs one.
