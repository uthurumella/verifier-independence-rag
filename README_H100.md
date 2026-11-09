# Running the full-scale poisoning on one H100 (80 GB)

Same models, prompts, checks and settings as the pilot. Only the backend (a local vLLM server instead of
Groq) and the speed (many questions at once) change.

| Attack | Script | Model | max_attempts |
|---|---|---|---|
| Contradiction | `src/data/attacks/contradiction_llm.py` | `openai/gpt-oss-20b` | 2 |
| Paraphrase | `src/data/attacks/paraphrase_llm.py` | `openai/gpt-oss-120b` | 3 |
| Highly relevant | `src/data/attacks/relevant_llm_r4.py --derived_forms` | `openai/gpt-oss-120b` | 3 |

Factual poisoning is rule-based and already full scale (`data/processed/poisoned_corpus/factual_v1`).

## What was changed
- `src/data/attacks/contradiction_llm.py`: `LocalClient` for an OpenAI-compatible server; new flags `--base_url`,
  `--workers`, `--request_timeout`, `--no_json_mode`, `--api_key_env`; `--reasoning_effort off`. Questions run in
  parallel; one thread writes the checkpoint; output order follows the seeded question order. Without
  `--base_url` the Groq path is unchanged.
- `src/data/attacks/relevant_llm.py`: the "current question" state is per-thread so parallel questions cannot mix.
- `scripts/run_poison_h100.sh`: starts each vLLM server, smoke-tests, runs the full attack, stops the server.
- `configs/index_config_full.yaml`, `configs/rag_config_full.yaml`:
  copies of the originals with the model revisions pinned to the pilot's commits and the variant list pointing
  at `*_llm_full_v1`. Nothing else differs; the run lock hash is unchanged. The original configs are untouched.
- `requirements.txt`: `torch` and `transformers` added (the code already imported them).
- Prompts, validation checks, prompt hashes (contradiction 2ba80bc687, paraphrase 8547a126ef) are unchanged.

## Setup
1. Project environment (Python 3.10+, CUDA torch): `pip install -r requirements.txt`
2. vLLM in a **separate** environment (it pins its own torch): `uv pip install vllm --torch-backend=auto`.
   Then `export VLLM_BIN=/path/to/that/env/bin/vllm`.
3. Hugging Face access to download `openai/gpt-oss-20b`, `openai/gpt-oss-120b` (about 65 GB), the e5 model
   and `Qwen/Qwen2.5-3B-Instruct`.
4. Data (not in git): `python src/data/load_hotpotqa.py`, then `python src/data/build_clean_corpus.py`.
   Splits are committed (`data/processed/splits`); expect 5,924 dev and 1,481 test.
5. Count the work first: `python src/data/attacks/contradiction_llm.py --split all --dry_run`

## Run
```
bash scripts/run_poison_h100.sh attacks     # per attack: smoke test (10 dev questions), then the full run
bash scripts/run_poison_h100.sh corpora
bash scripts/run_poison_h100.sh index
```
Look at `smoke_runs/<attack>/inspect.md` before trusting the full run. Output folders:
`data/processed/poisoned_corpus/{contradiction,paraphrase,relevant}_llm_full_v1`. Logs: `logs/h100/`.

Useful settings: `WORKERS=32`, `SPLIT=all`, `ATTACKS="contradiction"`, `SKIP_SMOKE=1`.
If the smoke test fails: HTTP 400 naming `reasoning_effort` -> `EXTRA_ARGS="--reasoning_effort off"`;
every reply "response_not_json" -> `EXTRA_ARGS="--no_json_mode"`. The 120b server needs vLLM's low-memory flags
on one 80 GB card; they are set in the script (override with `VLLM_120B_FLAGS`).

Stopping is safe: everything finished is in `checkpoint.jsonl`; re-run the same command to resume.
Use the `*_full_v1` folders only for this run: reusing a pilot folder name would mix Groq and local records.

## Then the RAG runs
The clean-corpus answers do not depend on the attacks, so reuse the finished `results/rag/clean_full` from Drive
(copy it into `results/rag/`); its run lock is the same as the pinned configs here.
```
python src/rag/run_rag.py --config configs/rag_config_full.yaml --tag full
python src/rag/build_clean_correct.py --config configs/rag_config_full.yaml
python src/rag/poison_on_clean_correct.py --config configs/rag_config_full.yaml --tag poison_on_clean_correct_full
```
Keep `batch_size: 16`. Use a new tag for every run and do not compare answers byte for byte with the T4 runs.
The last two commands overwrite `data/processed/clean_correct_top5_v1` and `poison_misled_cases_v1` (the pilot
versions are in git history); work on a branch or back them up first.

## Known limits
- The LLM-attack path was tested against a fake OpenAI-compatible server, not a real vLLM or gpt-oss.
- Generated text will differ from the Groq pilot (different serving stack, temperature 0.7); the accepted
  share of questions may differ too.
- The pilot's manual removals (3 paraphrase, 5 highly relevant questions) are not repeated; read a random
  sample of `inspect.md` instead.
