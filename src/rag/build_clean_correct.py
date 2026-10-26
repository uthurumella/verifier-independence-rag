"""Step 1 of 2: build the clean-correct dataset (no GPU, no model).

Reads a finished clean-corpus RAG run (default results/rag/clean_full, all 7,405
questions answered by run_rag.py) and keeps the questions the generator answered
correctly (F1 >= correct_f1 in the RAG config), each with the top-k passages it
answered from.

Output: data/processed/clean_correct_top5_v1/
    data.jsonl      one row per question: question, gold answer, generated answer,
                    F1/EM, top-k passages (rank, id, title, full text, gold flag),
                    and which attacks poisoned this question
    manifest.json   counts, source run, settings lock, field descriptions

Next step: python src/rag/poison_on_clean_correct.py

Usage:
    python src/rag/build_clean_correct.py
    python src/rag/build_clean_correct.py --clean_tag clean_full
"""

import argparse
import datetime as dt
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "retrieval"))
import index_common as ic  # noqa: E402
import smoke_test as st  # noqa: E402
import run_rag as R  # noqa: E402

CLEAN_SET = "clean_correct_top5_v1"


def require(paths, hint):
    """Stop with a readable message if an input file is missing."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit("Missing input file(s):\n  " + "\n  ".join(missing) + f"\n{hint}")


def passage_list(ids, passages, gold_ids, fake_ids=None, own_fake_ids=None):
    """Top-k passages in rank order with full text; fake flags only for poisoned corpora."""
    out = []
    for rank, pid in enumerate(ids, start=1):
        p = passages[pid]
        row = {"rank": rank, "passage_id": pid, "title": p["title"], "text": p["text"], "is_gold": pid in gold_ids}
        if fake_ids is not None:
            row["is_fake"] = pid in fake_ids
            row["fake_for_this_question"] = pid in own_fake_ids
        out.append(row)
    return out


def settings_lock(rcfg, icfg, clean_meta):
    """Lock hash of the current config with the model revisions of the clean run.
    Must equal the clean run's own lock: same retriever, top-k, generator and prompt."""
    lock = R.run_lock(rcfg, icfg, clean_meta["lock"]["generator"]["revision"], clean_meta["lock"]["dense"]["revision"])
    return lock, R.lock_hash(lock)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=R.DEFAULT_RAG_CONFIG)
    ap.add_argument("--clean_tag", default="clean_full", help="Finished clean-corpus run (results/rag/<clean_tag>).")
    ap.add_argument("--data_root", default="data/processed")
    args = ap.parse_args()

    require([args.config], "Run from the repository root.")
    rcfg = R.load_yaml(args.config)
    icfg = ic.load_config(rcfg["index_config"])
    clean_v = icfg["clean_variant"]
    top_k = rcfg["retrieval"]["top_k"]
    correct_f1 = rcfg["scoring"]["correct_f1"]

    clean_dir = os.path.join(rcfg["results_root"], args.clean_tag)
    require([os.path.join(clean_dir, "run_meta.json"), os.path.join(clean_dir, "per_question.jsonl")],
            f"Run the clean-corpus RAG first, e.g.:\n  python src/rag/run_rag.py --config "
            f"configs/rag_config_clean_full.yaml --tag {args.clean_tag} --variants none")
    require([icfg["paths"]["raw_dev"], icfg["paths"]["question_map"], ic.clean_passages_path(icfg)]
            + [ic.provenance_path(icfg, v) for v in icfg["variants"]],
            "Build the Task 1 data first (data/raw, data/processed: clean corpus, splits, poisoned corpora).")

    clean_meta = ic.read_json(os.path.join(clean_dir, "run_meta.json"))
    _, lock_h = settings_lock(rcfg, icfg, clean_meta)
    assert lock_h == clean_meta["lock_hash"], (
        f"config differs from the clean run {args.clean_tag} ({lock_h} vs {clean_meta['lock_hash']})")

    questions = st.load_questions(icfg)
    gold = st.load_gold(icfg)
    split_of = st.load_split_of(icfg)
    attacks = {v: st.load_attack(icfg, v) for v in icfg["variants"]}

    clean_rows = {r["question_id"]: r for r in ic.read_jsonl(os.path.join(clean_dir, "per_question.jsonl"))
                  if r["corpus"] == clean_v}
    assert clean_rows, f"no clean-corpus rows in {clean_dir}"
    correct_ids = sorted(q for q, r in clean_rows.items() if r["f1"] >= correct_f1)
    R.log(f"clean run {args.clean_tag}: {len(clean_rows)} questions, {len(correct_ids)} correct")

    needed = {p for q in correct_ids for p in clean_rows[q]["context_ids"]}
    passages = R.load_passages(icfg, [], needed)

    rows = [{
        "question_id": q, "split": split_of.get(q, "unknown"), "question_type": questions[q]["type"],
        "question": questions[q]["question"], "gold_answer": questions[q]["answer"],
        "generated_answer": clean_rows[q]["answer"], "f1": clean_rows[q]["f1"], "em": clean_rows[q]["em"],
        "n_gold_in_top_k": sum(x in gold[q] for x in clean_rows[q]["context_ids"]),
        "poisoned_by": [v for v in icfg["variants"] if q in attacks[v]],
        "top_k_passages": passage_list(clean_rows[q]["context_ids"], passages, gold[q]),
    } for q in correct_ids]

    out = os.path.join(args.data_root, CLEAN_SET)
    os.makedirs(out, exist_ok=True)
    R.write_jsonl(os.path.join(out, "data.jsonl"), rows)
    ic.write_json({
        "name": CLEAN_SET, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "description": f"Questions the RAG pipeline answered correctly (F1 >= {correct_f1}) on the clean corpus, "
                       f"with the top-{top_k} retrieved passages (full text, rank order) the answer was generated from.",
        "source_run": clean_dir, "source_lock_hash": lock_h,
        "source_per_question_sha256": ic.file_sha256(os.path.join(clean_dir, "per_question.jsonl")),
        "retriever": clean_meta["dense_model"], "generator": clean_meta["generator_model"], "top_k": top_k,
        "correct_f1": correct_f1, "n_rows": len(rows), "n_clean_questions": len(clean_rows),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_question_type": dict(Counter(r["question_type"] for r in rows)),
        "n_poisoned_by_attack": {v: sum(v in r["poisoned_by"] for r in rows) for v in icfg["variants"]},
        "fields": {"question_id": "HotpotQA id", "split": "dev or test", "question_type": "bridge or comparison",
                   "question": "question text", "gold_answer": "HotpotQA answer",
                   "generated_answer": "generator answer on the clean corpus", "f1": "F1 vs gold", "em": "exact match",
                   "n_gold_in_top_k": "gold supporting passages among the top-k",
                   "poisoned_by": "attacks that poisoned this question",
                   "top_k_passages": "rank, passage_id, title, text, is_gold"},
    }, os.path.join(out, "manifest.json"))
    R.log(f"done: {CLEAN_SET} {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
