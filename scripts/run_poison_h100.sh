#!/usr/bin/env bash
# Full-scale LLM poisoning on one H100 with locally served models (vLLM). No Groq, no API key.
#
# Same models, prompts, checks and settings as the pilot; only the backend and the speed change.
#   contradiction : openai/gpt-oss-20b   (max_attempts 2, the script default)
#   paraphrase    : openai/gpt-oss-120b  (max_attempts 3)
#   highly relevant: openai/gpt-oss-120b (max_attempts 3, relevant_llm_r4.py --derived_forms)
#
# Usage (from anywhere inside the repo):
#   bash scripts/run_poison_h100.sh attacks     # servers + 10-question smoke test + full run, per attack
#   bash scripts/run_poison_h100.sh corpora     # clean corpus + each attack's passages -> corpus.jsonl
#   bash scripts/run_poison_h100.sh index       # build indexes and run the retrieval smoke test (GPU)
#   bash scripts/run_poison_h100.sh all         # the three stages in order
#
# Re-running a stage resumes from the checkpoints. Stages never delete anything.
#
# Settings (environment variables, all optional):
#   WORKERS=32            questions processed at once
#   SPLIT=all             dev | test | all
#   LIMIT=100000          max eligible questions per attack
#   ATTACKS="contradiction paraphrase relevant"   which attacks to run
#   SMOKE_N=10            questions in the smoke test; SMOKE_MIN_ACCEPTED=1 needed to continue
#   SKIP_SMOKE=1          skip the smoke test
#   PORT=8000             vLLM port
#   VLLM_BIN=vllm         vLLM command (use a full path if it lives in another virtualenv)
#   PYTHON=python         the project's Python (needs requests, torch, ...; NOT the vLLM environment)
#   EXTRA_ARGS=""         extra flags for the attack scripts, e.g. "--reasoning_effort off" or "--no_json_mode"
#   VLLM_20B_FLAGS / VLLM_120B_FLAGS   replace the default vLLM flags
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGE="${1:-all}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-32}"
SPLIT="${SPLIT:-all}"
LIMIT="${LIMIT:-100000}"
ATTACKS="${ATTACKS:-contradiction paraphrase relevant}"
SMOKE_N="${SMOKE_N:-10}"
SMOKE_MIN_ACCEPTED="${SMOKE_MIN_ACCEPTED:-1}"
VLLM="${VLLM_BIN:-vllm}"
PY="${PYTHON:-python}"
EXTRA="${EXTRA_ARGS:-}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
ROOT="data/processed/poisoned_corpus"
LOGDIR="logs/h100"
# the 120b model is ~65 GB (MXFP4): on one 80 GB H100 vLLM needs these (from the vLLM gpt-oss recipe)
FLAGS_20B="${VLLM_20B_FLAGS:---max-model-len 16384 --max-num-seqs 64}"
FLAGS_120B="${VLLM_120B_FLAGS:---gpu-memory-utilization 0.95 --max-num-batched-tokens 1024 --max-model-len 16384 --max-num-seqs 64}"

mkdir -p "$LOGDIR" smoke_runs
SERVER_PID=""

log() { echo "[$(date +%H:%M:%S)] $*"; }

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "stopping vLLM (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT

start_server() {  # model, flags
  local model="$1" flags="$2"
  if curl -sf "$BASE_URL/models" >/dev/null 2>&1; then
    echo "Port $PORT already has a server running. Stop it first (or set PORT)." >&2; exit 1
  fi
  log "starting vLLM: $model  ($flags)"
  # shellcheck disable=SC2086
  "$VLLM" serve "$model" --port "$PORT" $flags > "$LOGDIR/vllm_$(basename "$model").log" 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 480); do   # up to 40 minutes: the first start downloads the weights
    if curl -sf "$BASE_URL/models" >/dev/null 2>&1; then log "server ready"; return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM exited early. Last log lines:" >&2; tail -25 "$LOGDIR/vllm_$(basename "$model").log" >&2; exit 1
    fi
    sleep 5
  done
  echo "vLLM did not become ready in 40 minutes. See $LOGDIR" >&2; exit 1
}

accepted_count() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['questions_accepted'])" "$1"; }

# run_attack NAME SCRIPT MODEL MAX_ATTEMPTS [extra flags for this attack]
run_attack() {
  local name="$1" script="$2" model="$3" attempts="$4"; shift 4
  local out="$ROOT/${name}_llm_full_v1"
  if [[ -z "${SKIP_SMOKE:-}" ]]; then
    log "[$name] smoke test: $SMOKE_N dev questions -> smoke_runs/$name"
    # shellcheck disable=SC2086
    "$PY" "$script" --base_url "$BASE_URL" --model "$model" --max_attempts "$attempts" \
      --split dev --limit "$SMOKE_N" --workers 8 --out_dir "smoke_runs/$name" "$@" $EXTRA \
      | tee "$LOGDIR/smoke_$name.log" | tail -15
    local ok; ok="$(accepted_count "smoke_runs/$name/run_manifest.json")"
    if (( ok < SMOKE_MIN_ACCEPTED )); then
      echo "[$name] smoke test accepted $ok questions (need $SMOKE_MIN_ACCEPTED). Read smoke_runs/$name/inspect.md and $LOGDIR/smoke_$name.log." >&2
      echo "If the server rejects 'reasoning_effort' re-run with EXTRA_ARGS='--reasoning_effort off'; if every reply is not JSON use EXTRA_ARGS='--no_json_mode'." >&2
      exit 1
    fi
  fi
  log "[$name] full run: split=$SPLIT workers=$WORKERS -> $out"
  # shellcheck disable=SC2086
  "$PY" "$script" --base_url "$BASE_URL" --model "$model" --max_attempts "$attempts" \
    --split "$SPLIT" --limit "$LIMIT" --workers "$WORKERS" --out_dir "$out" "$@" $EXTRA \
    | tee -a "$LOGDIR/full_$name.log" | tail -25
  local stopped
  stopped="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('stopped_early') or '')" "$out/run_manifest.json")"
  if [[ -n "$stopped" ]]; then
    echo "[$name] run stopped early: $stopped" >&2
    echo "Everything finished is checkpointed. Fix the cause and re-run the same command to resume." >&2
    exit 1
  fi
}

want() { [[ " $ATTACKS " == *" $1 "* ]]; }

stage_attacks() {
  "$PY" -c "import requests" || { echo "The project Python needs 'requests' (pip install -r requirements.txt)." >&2; exit 1; }
  command -v "$VLLM" >/dev/null || { echo "vLLM not found ('$VLLM'). Install it in its own environment and set VLLM_BIN." >&2; exit 1; }
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  [[ -f data/raw/dev.jsonl ]] || { echo "data/raw/dev.jsonl missing: run src/data/load_hotpotqa.py first." >&2; exit 1; }

  if want contradiction; then
    start_server openai/gpt-oss-20b "$FLAGS_20B"
    run_attack contradiction src/data/attacks/contradiction_llm.py openai/gpt-oss-20b 2
    stop_server
  fi
  if want paraphrase || want relevant; then
    start_server openai/gpt-oss-120b "$FLAGS_120B"
    want paraphrase && run_attack paraphrase src/data/attacks/paraphrase_llm.py openai/gpt-oss-120b 3
    want relevant && run_attack relevant src/data/attacks/relevant_llm_r4.py openai/gpt-oss-120b 3 --derived_forms
    stop_server
  fi
  log "attacks done. Next: bash scripts/run_poison_h100.sh corpora"
}

stage_corpora() {
  [[ -f data/processed/clean_corpus/passages.jsonl ]] || { echo "clean corpus missing: run src/data/build_clean_corpus.py first." >&2; exit 1; }
  for name in contradiction paraphrase relevant; do
    want "$name" || continue
    local dir="$ROOT/${name}_llm_full_v1"
    [[ -f "$dir/poisoned_passages.jsonl" ]] || { echo "$dir/poisoned_passages.jsonl missing: run the attacks stage first." >&2; exit 1; }
    "$PY" src/data/build_poisoned_corpus.py --poison_path "$dir/poisoned_passages.jsonl" \
      --out_dir "$dir" --variant_name "${name}_llm_full_v1"
  done
  # the factual attack (rule-based, already full scale) also needs its corpus on a fresh machine
  if [[ ! -f "$ROOT/factual_v1/corpus.jsonl" && -f "$ROOT/factual_v1/poisoned_passages.jsonl" ]]; then
    "$PY" src/data/build_poisoned_corpus.py --poison_path "$ROOT/factual_v1/poisoned_passages.jsonl" \
      --out_dir "$ROOT/factual_v1" --variant_name factual_v1
  fi
}

stage_index() {
  "$PY" src/retrieval/build_indexes.py --config configs/index_config_full.yaml
  "$PY" src/retrieval/smoke_test.py --config configs/index_config_full.yaml --tag full
}

case "$STAGE" in
  attacks) stage_attacks ;;
  corpora) stage_corpora ;;
  index)   stage_index ;;
  all)     stage_attacks; stage_corpora; stage_index ;;
  *) echo "usage: $0 attacks|corpora|index|all" >&2; exit 2 ;;
esac
log "stage '$STAGE' finished"
