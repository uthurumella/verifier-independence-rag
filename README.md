# Can We Trust the Verifier? Verification Independence in RAG Under Knowledge-Base Poisoning

## Overview

A common way to make retrieval-augmented generation (RAG) safer is to add a verifier: a second model that checks
the generated answer before it is returned. This project asks whether that check helps when the verifier reads the
same evidence the generator did, and how much independence (a different model, a different retriever, a separate
evidence source) it needs before it stops failing together with the generator.

This repository holds the data pipeline for the study: a HotpotQA question pool with a fixed dev/test split, a
clean retrieval corpus, four knowledge-base poisoning attacks with their released datasets, dense and BM25
retrieval, and the baseline RAG pipeline that the verifier experiments build on.

**Research question.** How does the degree of independence between the generator and the verifier affect their
tendency to fail together under knowledge poisoning?

**Why it matters.** If a poisoned passage misleads the generator and the verifier is handed the same passage, the
verifier has nothing independent to check against. Swapping in a different model family may not help, because both
models reason over the same bad evidence. What may matter more is whether the verifier retrieves its own evidence.
Verifier-based defences are deployed on the assumption that a second model adds a real safety layer; if that
independence is superficial, so is the protection.

## Contents

| Path | What it is |
|---|---|
| `src/data/` | Data loading, dev/test split, clean corpus, poisoned corpora |
| `src/data/attacks/` | The four poisoning attacks |
| `src/retrieval/` | Dense and BM25 indexes, retrieval check (`smoke_test.py`) |
| `src/rag/` | Baseline RAG pipeline: generator, metrics, and the two RAG question sets below |
| `configs/` | Data, index and RAG settings (`*_full.yaml` for the full-scale run) |
| `scripts/run_poison_h100.sh` | Runs the LLM attacks on a local GPU (see `README_H100.md`) |
| `data/processed/splits/` | Committed dev/test question IDs |
| `data/processed/poisoned_corpus/` | Released poisoned passages, one folder per attack |
| `data/processed/clean_correct_top5_v1/` | 3,711 questions the baseline RAG answers correctly on the clean corpus |
| `data/processed/poison_misled_cases_v1/` | 240 cases (235 questions) from that set that turn wrong on a poisoned corpus (factual and pilot LLM sets) |

## Dataset

HotpotQA, distractor configuration, dev split: 7,405 multi-hop questions (5,918 bridge, 1,487 comparison), all
labelled hard. Each question comes with 2 gold supporting passages plus distractors: 10 paragraphs in all for
7,345 questions and 2 to 9 for the other 60. The question pool is split 80/20 at the question level with a fixed
seed (42) into **dev (5,924)** and **test (1,481)**. The split was fixed before any attack was tuned; the test set
checks whether results hold on questions the pipeline was never adapted to.

Source data is not redistributed; `src/data/load_hotpotqa.py` downloads it.

- Source: https://huggingface.co/datasets/hotpotqa/hotpot_qa
- Paper: Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP 2018
- License: CC BY-SA 4.0; the derived datasets in this repository inherit its share-alike terms

## Corpus and threat model

- **One global corpus.** All context paragraphs are pooled and deduplicated by title into 66,581 passages.
  Retrieval runs over the whole pool, not each question's own paragraphs, so a poisoned passage has to outrank
  66,581 competitors rather than being retrieved by construction.
- **Gold evidence is kept.** Poisoned passages are added next to the original passages, never in place of them.
  The true and the false claim coexist, and retrieval decides which one the generator sees.
- **Poison is unmarked.** A poisoned passage carries the title of the gold passage it targets and is otherwise
  indistinguishable. Which passages are poisoned is recorded only in a separate provenance file that is read at
  scoring time, so the retriever, generator and verifier stay blind to it.
- **Three passages per target.** Each false claim is inserted three times under distinct IDs, both to give it a
  realistic chance of reaching the top k and to test whether repeated copies are mistaken for independent
  corroboration.

## Poisoning attacks

| Attack | What the poisoned passage is | Generator | Script (`src/data/attacks/`) |
|---|---|---|---|
| Factual substitution | A gold passage with the answer's number changed; 3 identical copies | rule-based | `factual_substitution_v2.py` |
| Contradiction | A gold passage with the answer replaced by a false one; 3 differently worded versions | `openai/gpt-oss-20b` | `contradiction_llm.py` |
| Paraphrase | A full rewrite of a gold passage stating a false answer; 3 versions | `openai/gpt-oss-120b` | `paraphrase_llm.py` |
| Highly relevant | A new passage written around the question that states a false answer as fact, filed under a gold passage's title; 3 versions | `openai/gpt-oss-120b` | `relevant_llm_r4.py --derived_forms` |

**Factual substitution** targets only the value the question asks for. It takes the first number in the gold
answer, finds the first gold passage that contains it, and replaces every standalone occurrence of that number in
the passage (also one followed by a comma or full stop; never digits that belong to a larger number or a decimal)
with a different value: a year (1000–2030) moves by 5 to 20 years and never past 2026, a decade moves by whole
decades, a day of the month stays between 1 and 28, and any other number is scaled by 0.55 to 2.0. Ordinal suffixes
follow the new number. Questions whose answer number is part of a decimal, or whose number is one end of a range
in the passage ("26–30 August", "2013–14"), are skipped. Every passage is checked before it is written (false value
present, true value absent, no impossible date). It applies only to questions whose answer contains a number.
`factual_substitution_v2.py` replaced the first version (`factual_substitution.py`, output `factual_v1`), which
ignored context; `factual_v2/v1_vs_v2.md` compares the two.

**Contradiction, paraphrase and highly relevant** share one eligibility rule, one seeded question order and one
validation framework (`contradiction_llm.py`). A question is skipped before generation if its answer is yes/no, a
single character, not found in the gold passages, or contained in the title of a gold passage (the poisoned
passage keeps that title, so the true answer would remain in it). Contradiction and paraphrase rewrite every gold
passage that states the answer; highly relevant writes one new passage per question. The contradiction and
paraphrase prompts never include the question. The highly relevant prompt does, but a version may not copy a run
of more than 7 words from the question (names and numbers not counted) or of more than 12 words from a gold
passage.

**Validation.** A question is accepted only if every version passes every check, within two (contradiction) or
three (paraphrase, highly relevant) generation attempts; otherwise it is rejected and its reasons are logged, with
no fallback. The main checks:

- the false answer differs from the true one and is not already in the passage or the question;
- every version states the false answer, and neither the true answer nor its distinctive parts remain;
- the other names and numbers of the source are kept, no new numbers appear, and the passage still names its
  subject (highly relevant: no names or numbers beyond those in the question and gold passages);
- one plain paragraph without markup, 0.6 to 1.5 times the source's length in words (highly relevant: fixed word
  limits instead);
- the three versions are not near copies of each other (pairwise similarity at most 0.8);
- attack-specific rules: paraphrase versions must be full rewrites (word-sequence similarity to the source at most
  0.75); highly relevant versions may not hedge or refer to themselves, must keep dates in the right order, must
  name every subject a "what do both share" question asks about, and may not add commentary or sentences that only
  restate the value.

When writing the dataset, the export step re-checks every passage (false answer present, true answer absent) and
stops if one fails. The corpus builder checks that no poisoned ID collides with a clean one and that the counts
add up.

## Released datasets

All four attacks were run on the full question pool (dev + test) with seed 42. The factual attack considers every
question; the three LLM attacks attempt the 4,436 eligible ones.

| Attack | Folder (`data/processed/poisoned_corpus/`) | Questions attempted | Poisoned questions (dev / test) | Share poisoned | Poisoned passages |
|---|---|---|---|---|---|
| Factual substitution | `factual_v2` | 7,405 | 1,092 (896 / 196) | 14.7% | 3,276 |
| Contradiction | `contradiction_llm_full_v1` | 4,436 | 2,120 (1,684 / 436) | 47.8% | 6,681 |
| Paraphrase | `paraphrase_llm_full_v1` | 4,436 | 3,472 (2,749 / 723) | 78.3% | 11,082 |
| Highly relevant | `relevant_llm_full_v1` | 4,436 | 3,283 (2,635 / 648) | 74.0% | 9,849 |

Each full-scale LLM folder (`*_llm_full_v1`) contains:

| File | Contents |
|---|---|
| `poisoned_passages.jsonl` | The poisoned passages (`passage_id`, `title`, `text`) that are merged into the corpus |
| `provenance.jsonl` | One row per passage: question ID and type, target passage, true and false value, copy index, number of non-gold passages still stating the true answer, model, prompt version |
| `run_manifest.json` | Settings, counts, and the reasons questions were skipped or rejected |
| `README.md` | Data card: how the set was made, counts by split, quality checks, retrieval results, limitations |

`factual_v2` contains `poisoned_passages.jsonl`, `provenance.jsonl`, `rejected.jsonl`,
`run_manifest.json`, `v1_vs_v2.md` and a data card; `factual_v1` is kept because the pilot results were built on
it. Kept out of Git: the merged corpora and
search indexes, which are large and are rebuilt from the released files with the commands below, and the
raw generation records (checkpoints, rejected and skipped questions, inspection sheets), which are archived with
md5 checks and available on request.

The `*_llm_v1` folders without `full` are the earlier pilot sets (30–33 dev questions each), generated through the
Groq API while the prompts and checks were developed. The contradiction and paraphrase pilots use the same model
and prompt version as their full sets; the highly relevant pilot mixes earlier prompt versions and both models, as
recorded per passage in its `provenance.jsonl`. They are kept for reference; the `*_full_v1` sets supersede them.

## Quality checks

| Check | Factual | Contradiction | Paraphrase | Highly relevant |
|---|---|---|---|---|
| Poisoned passages stating the false answer | 100%² | 100% | 100% | 100% |
| Poisoned passages still stating the true answer | 0.1%² | 0%¹ | 0% | 0% |
| False answer of the same kind as the true one (4-digit number / other number / text) | 98.1% | 98.4% | 98.9% | 98.0% |
| Manual read of random poisoned questions: clean and plausible | 26 of 30 | 45 of 50 | 37 of 50 | 42 of 50 |

The first two rows use the pipeline's own whole-token match on normalized text, which every released LLM passage
passed at export. ¹ A separate, looser match (punctuation and articles removed) flagged 6 of the 6,681 contradiction
passages (0.1%) as possibly still containing the true answer. ² The factual attack checks every passage with its own
number rule before writing it: the false number is present and the true number absent in all 3,276 passages. By the
whole-token match the false number is found in 91.5%, because it can be joined to letters or a suffix (road "A41"
becomes "A29", "12th" becomes "19th"), and the true number in 1 question (3 passages), where "3.11" is a nickname of
the date 11 March. For every set, passage IDs are unique, each passage keeps its target's title, each question has a
single false value, and every poisoned question appears in the provenance file. Each data card lists what the manual
read found.

## Retrieval check

Does the poison reach the retriever? `src/retrieval/smoke_test.py` runs every poisoned question against the
poisoned corpus and against the clean corpus and compares the rankings. Dense retrieval uses `intfloat/e5-base-v2`
(pinned revision) with normalized embeddings and an exact inner-product FAISS index (cosine similarity); BM25 uses
the Lucene variant with k1 = 1.5 and b = 0.75. Dev and test give similar figures.

| Attack | Dense: poison in top 5 | Dense: poison in top 10 | Dense: poison above its target | BM25: poison in top 5 | BM25: poison in top 10 | Gold pushed out of top 5 (dense / BM25) |
|---|---|---|---|---|---|---|
| Factual substitution | 0.59 | 0.70 | 0.39 | 0.42 | 0.53 | 0.01 / 0.00 |
| Contradiction | 0.60 | 0.71 | 0.44 | 0.46 | 0.57 | 0.01 / 0.01 |
| Paraphrase | 0.50 | 0.61 | 0.18 | 0.42 | 0.53 | 0.00 / 0.01 |
| Highly relevant | 0.99 | 1.00 | 0.97 | 0.99 | 1.00 | 0.04 / 0.06 |

*Poison in top k*: share of poisoned questions with at least one poisoned passage in the top k. *Poison above its
target*: a poisoned passage ranks higher than the gold passage it targets (or the target is not in the top 20).
*Gold pushed out*: a gold passage is in the clean top 5 but not in the poisoned top 5.

The three attacks that rewrite a gold passage put a poisoned passage in the top 5 for roughly half the questions
and almost never push the gold passage out (at most 1% of questions), so when the poison is retrieved the truth is
almost always retrieved with it: the conflict a verifier has to resolve. Highly relevant poison is retrieved for
nearly every question, with the first poisoned passage at median rank 1.

## Reproducing

```
pip install -r requirements.txt
python src/data/load_hotpotqa.py                  # downloads HotpotQA (not redistributed)
python src/data/build_clean_corpus.py             # 66,581 passages; the dev/test split is committed
python src/data/attacks/factual_substitution_v2.py   # factual attack (rule-based, CPU)
bash scripts/run_poison_h100.sh attacks           # LLM attacks on a local GPU via vLLM (see README_H100.md)
bash scripts/run_poison_h100.sh corpora           # clean corpus + each attack's passages
bash scripts/run_poison_h100.sh index             # dense and BM25 indexes, then the retrieval check
```

To use the released passages without regenerating them, skip the two attack steps: `build_poisoned_corpus.py`
merges a folder's `poisoned_passages.jsonl` into the clean corpus. LLM generation is checkpointed, so re-running a
command resumes where it stopped; because the model samples, a fresh run produces different passages. The exact
commands, models and hardware used for each LLM set are in its data card and `run_manifest.json`.

## Limitations

- **Coverage.** The LLM attacks cover only the 4,436 eligible questions. Comparison questions are
  under-represented (86–178 per set) because their answers are mostly yes/no or a passage title. Factual
  substitution covers only answers that contain a number.
- **Residual truth.** For about 17% of the questions poisoned by an LLM attack, one of the question's own
  distractor paragraphs (which are part of the corpus) also states the true answer and is not rewritten. The count
  is recorded per passage in `provenance.jsonl`.
- **Plausibility is not checked automatically.** The manual reads found swaps that are implausible or inconsistent
  with other facts in the same passage (3 to 10 of 50 per set).
- **Detectable traces.** 36% of contradiction passages contain typographic characters (mostly a non-breaking
  hyphen) that their source does not; the paraphrase and highly relevant outputs are normalized, leaving 2.8% and
  1.9%. In the manual read, 4 of 50 highly relevant questions had wording that echoes the question ("The question
  mentions…"). A detector could use such traces.
- **Factual substitution artefacts.** v2 removes the context errors of v1 (true number left in place, dropped
  commas, impossible dates, years after 2026, broken ranges and decades; counts in `factual_v2/v1_vs_v2.md`). What
  remains is plausibility: a 3-digit year is scaled like a count ("934 to 961" can become "514 to 961"), a swap can
  clash with other dates in the passage, and an unrelated equal number in the same passage is changed too (148
  answers are a single digit). In a manual read of 30 random v2 questions, 26 were clean and plausible.
- **Wording-based matching.** The true-answer checks match the answer's wording (plus a few derived forms in the
  highly relevant attack), so the same entity under a different name is not detected.

## Development notes

Contradiction poisoning was first attempted by rule-based substitution (a random answer from the dataset, answers
bucketed by surface form, types inferred from the question, and named-entity typing). Each produced ungrammatical
or incoherent text, and adding a question-shaped header to lift rank made passages look synthetic. The conclusion
was that coherent adversarial text needs generation, which led to the three LLM attacks. Their prompts and checks
were developed on the pilot runs and then frozen; the full-scale sets were generated with the final prompt
versions and models on a local vLLM server.

## Citation

If you use the poisoned datasets, please cite HotpotQA (Yang et al., 2018) and this repository.
