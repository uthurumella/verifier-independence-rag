# contradiction_llm_full_v1 — full-scale contradiction poisoning (HotpotQA, dev + test)

Knowledge-base poisoning set for the project "Can We Trust the Verifier? Verification Independence in RAG Under
Knowledge-Base Poisoning". Each poisoned question gets fake versions of each gold passage that states the answer,
in which the answer is replaced by a false value. Built with the same prompt, model, checks and seed as the pilot
set `contradiction_llm_v1`; only the serving backend (a local vLLM server instead of the Groq API) and the scale
changed.

## What is in this folder

| File | Rows | In Git | What it is |
|---|---|---|---|
| `poisoned_passages.jsonl` | 6,681 | yes | the fake passages: `passage_id`, `title`, `text` |
| `provenance.jsonl` | 6,681 | yes | one row per fake passage: question ID and type, source passage, true value, false value, copy index, model, prompt version |
| `run_manifest.json` | — | yes | settings and summary counts |
| `README.md` | — | yes | this data card |
| `rejected.jsonl` | 2,316 | no (Drive) | eligible questions whose output failed the checks, with reasons |
| `skipped.jsonl` | 2,969 | no (Drive) | questions not eligible for this attack, with the reason |
| `checkpoint.jsonl` | 4,436 | no (Drive) | raw record per attempted question (source of truth; every other file is rebuilt from it) |
| `inspect.md` | — | no (Drive) | human-readable sheet: source passage and all versions for every question |
| `corpus.jsonl`, `corpus_manifest.json` | 73,262 | no (git-ignored) | clean corpus + fake passages; rebuild with `src/data/build_poisoned_corpus.py` |

Only the four "yes" files are in Git: the two data files that the corpus, retrieval, RAG and verifier scripts
read, plus the run manifest and this card. The "Drive" files are kept, with md5-checked backups, in the project's
Google Drive folder and are available on request. Search indexes (`data/indexes/contradiction_llm_full_v1`) and
the retrieval check (`results/smoke_test/full_contradiction`) are also kept on Drive; both can be rebuilt (see
below).

## How it was made

- Questions: HotpotQA distractor dev set, 7,405 questions, split with seed 42 into dev 5,924 / test 1,481
  (`data/processed/splits`).
- Script: `src/data/attacks/contradiction_llm.py` (this branch), prompt version `2ba80bc687`.
- Model: `openai/gpt-oss-20b` served with vLLM 0.34.0 on one NVIDIA L4 (Colab Pro), 32 parallel requests.
- Settings: seed 42, `--split all`, `max_attempts` 2, 3 versions per target passage, maximum pairwise similarity
  between versions 0.8.
- Run dates: 2026-11-20. Dev was generated first (`--split dev`); the test split was added by re-running with
  `--split all` on the same folder, which reuses every finished dev record from the checkpoint (all 5,298 dev
  passages are byte-identical before and after).

Commands:
```
vllm serve openai/gpt-oss-20b --port 8000 --max-model-len 16384 --max-num-seqs 64
python src/data/attacks/contradiction_llm.py --base_url http://127.0.0.1:8000/v1 --model openai/gpt-oss-20b \
  --max_attempts 2 --split all --limit 100000 --workers 32 \
  --out_dir data/processed/poisoned_corpus/contradiction_llm_full_v1
python src/data/build_poisoned_corpus.py --poison_path data/processed/poisoned_corpus/contradiction_llm_full_v1/poisoned_passages.jsonl \
  --out_dir data/processed/poisoned_corpus/contradiction_llm_full_v1 --variant_name contradiction_llm_full_v1
python src/retrieval/build_indexes.py --config configs/index_config_full.yaml --variants contradiction_llm_full_v1
python src/retrieval/smoke_test.py --config configs/index_config_full.yaml --variants contradiction_llm_full_v1 --tag full_contradiction
```

## Counts

| | Dev | Test | Total |
|---|---|---|---|
| Questions | 5,924 | 1,481 | 7,405 |
| Not eligible | 2,377 | 592 | 2,969 |
| Eligible (attempted) | 3,547 | 889 | 4,436 |
| **Accepted (poisoned)** | **1,684 (47.5%)** | **436 (49.0%)** | **2,120 (47.8%)** |
| Rejected by the checks | 1,863 | 453 | 2,316 |
| Fake passages | 5,298 | 1,383 | 6,681 |

Not eligible (total): answer appears in the gold passage title 2,492; yes/no answer 458; answer too short 14;
answer absent from the gold passages 5.

Accepted: bridge 2,034, comparison 86; one target passage 2,013, two target passages 107 (2,227 target passages ×
3 versions = 6,681). For 350 accepted questions one of the question's own distractor paragraphs still states the
true answer (not rewritten).

Rejection reasons (a question can have several): versions too similar 1,436; a required name missing 1,429; fewer
than 3 versions returned 482; true answer still present 439; a required number missing 428; part of the true
answer present 289; false answer absent 280; subject dropped 250; length ratio 141; output truncated 140; passage
missing 119; number added 86; replacement already in source 72; other 58.

## Quality checks (2026-11-20)

- Structure: 6,681 passages = 6,681 provenance rows; all passage ids unique and distinct from clean ids; exactly 3
  versions per target passage; every fake keeps its source title; one false value per question; one model and
  prompt version throughout.
- Content: with the pipeline's own whole-token match on normalized text, which the export enforces, every fake
  passage states the false value and none states the true value. A separate, looser match (punctuation and
  articles removed) found the false value in 99.3% of fake passages (the rest join it to the previous word with a
  dash, e.g. "children—sixteen") and flagged 6 of 6,681 (0.1%, all dev) as possibly still containing the true
  value. Test part: false value present 1,383/1,383, true value 0/1,383 under both.
- False answer of the same kind as the true one (4-digit number / other number / text): 98.4%.
- Length relative to the source passage (words, as the length check counts them): median 1.00× (range 0.62–1.50×;
  the check allows 0.6–1.5×). Pairwise similarity between the 3 versions: mean 0.62, maximum 0.80.
- Manual audit, 50 random accepted dev questions: 45 clean and plausible; 2 weak near-synonym swaps (e.g.
  "Universal Studios Florida" → "Universal Studios Orlando"); 3 implausible (e.g. Guarani described as a language
  "of North America"); none states the true answer.
- Agreement with the pilot: 26 of the pilot's 33 accepted questions are accepted again (same false value in 9;
  generation uses temperature 0.7).

## Does the poison reach the retriever?

Retrieval check over all 2,120 poisoned questions (`smoke_test.py`, e5-base-v2 dense and BM25 indexes built with
`configs/index_config_full.yaml`):

| Retriever | fake in top 5 | fake in top 10 | gold in top 5: clean → poisoned | gold pushed out of top 5 | verdict |
|---|---|---|---|---|---|
| Dense (e5-base-v2, cosine) | 0.60 | 0.71 | 0.97 → 0.96 | 0.01 | surfaces |
| BM25 (Lucene, k1 1.5, b 0.75) | 0.46 | 0.57 | 0.95 → 0.94 | 0.01 | surfaces |

Dev and test give similar figures (dense poison@5 0.61 / 0.60; BM25 0.46 / 0.47). The fake passages appear next to
the true one rather than pushing it out of the top 5, but they often take first place: gold at rank 1 drops from
0.87 to 0.73 (dense) and from 0.82 to 0.75 (BM25), and a fake ranks above the passage it was made from (or that
passage is not in the top 20) in 44% (dense) / 26% (BM25) of questions. Full figures:
`results/smoke_test/full_contradiction/summary.json`.

## Known limitations

- Typography: 36% of fake passages contain characters that the source passage does not (mostly the non-breaking
  hyphen U+2011, also ’, —, narrow no-break space). The pilot set has the same pattern (38.9%, 42 of 108). Kept as
  generated; a detector could use it as a signal.
- The automatic checks do not test plausibility: a few swaps are implausible or internally inconsistent (see the
  audit above).
- 350 accepted questions (16.5%) keep a distractor paragraph that states the true answer, which may weaken the
  attack for them.
- Comparison questions are few (86), because most comparison answers are yes/no or a passage title.

## Rebuilding the corpus and indexes

`load_hotpotqa.py` → `build_clean_corpus.py` → `build_poisoned_corpus.py` → `build_indexes.py` → `smoke_test.py`,
with the commands above. No GPU is needed for the corpus; the dense index needs one.
