"""Task 3: generator-verifier independence.

The generator's answers stay fixed (results/rag/<rag_tag>/per_question.jsonl).
Only the evidence the verifier reads changes:

    same_evidence  the generator's own top-k passages (Task 2 results, reused)
    bm25           an independent retriever (BM25) over the SAME corpus the
                   generator used: clean, or that attack's poisoned corpus
    reference      dense retrieval over the separately indexed clean HotpotQA
                   corpus, which never contains poison (ranked lists reused from
                   the generator run's clean-corpus retrieval)

Each condition is judged by the same two verifier models as Task 2 (same
prompt, same Yes/No scoring), so a change of MODEL and a change of EVIDENCE can
be compared directly. For every answer it also records how many of the
verifier's top-k passages are the generator's (overlap) and whether the
verifier's evidence contains a poisoned passage.

Usage:
    python src/rag/verify_independent.py --limit 5 --tag check5   # quick check
    python src/rag/verify_independent.py                         # full run (resumable)
    python src/rag/verify_independent.py --score_only            # rebuild reports

Outputs in results/independence/<tag>/:
    run_meta.json, evidence.jsonl (verifier top-k per condition), verdicts.jsonl,
    per_row.jsonl, summary.json, summary.md, examples.md
"""

import argparse
import datetime as dt
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "retrieval"))
import index_common as ic  # noqa: E402
import smoke_test as st  # noqa: E402
import generator as G  # noqa: E402
import run_rag as R  # noqa: E402
import verify_same_evidence as V  # noqa: E402  (verifier models, prompt and metrics from Task 2)

DEFAULT_CONFIG = "configs/independence_config.yaml"
CONDITION_TEXT = {
    "same_evidence": "same evidence as the generator",
    "bm25": "independent retriever (BM25), same corpus",
    "reference": "separate reference view (clean index)",
}
log = V.log


class SplitOnOOM:
    """Wraps a verifier: a batch that does not fit in GPU memory is scored in halves instead
    (same prompts, same scores, only slower)."""

    def __init__(self, inner):
        self.inner = inner

    def prompt_text(self, messages):
        return self.inner.prompt_text(messages)

    def score(self, batch):
        try:
            return self.inner.score(batch)
        except Exception as error:  # torch.OutOfMemoryError without importing torch here
            if "out of memory" not in str(error).lower() or len(batch) == 1:
                raise
            V.free_gpu()
            half = len(batch) // 2
            return self.score(batch[:half]) + self.score(batch[half:])


# ---------------------------------------------------------------- evidence

def bm25_evidence(icfg, wanted, questions, k):
    """wanted: corpus -> [question ids]. BM25 top-k over that same corpus."""
    out = {}
    for corpus, qids in sorted(wanted.items()):
        index = ic.BM25Index.load(icfg, corpus)
        ranked, _ = index.search([questions[q]["question"] for q in qids], k)
        out.update({(corpus, q): r[:k] for q, r in zip(qids, ranked)})
        log(f"bm25 evidence: {corpus} {len(qids)} questions")
        del index
    return out


def reference_evidence(rag_dir, reference_variant, wanted, k):
    """Dense top-k over the clean reference index, from the generator run's cached rankings."""
    ranked = {r["question_id"]: r["ranked"] for r in ic.read_jsonl(os.path.join(rag_dir, "retrieval.jsonl"))
              if r["corpus"] == reference_variant}
    out = {}
    for corpus, qids in wanted.items():
        for q in qids:
            assert q in ranked, f"{q}: no {reference_variant} ranking in {rag_dir}/retrieval.jsonl"
            out[(corpus, q)] = ranked[q][:k]
    return out


def build_evidence(cfg, icfg, rag_dir, rows, questions, k, path):
    """Cached in evidence.jsonl: {condition, corpus, question_id, context_ids}."""
    have = {}
    if os.path.exists(path):
        for r in ic.read_jsonl(path):
            have[(r["condition"], r["corpus"], r["question_id"])] = r["context_ids"]
    for cond in cfg["conditions"]:
        wanted = {}
        for r in rows:
            if (cond, r["corpus"], r["question_id"]) not in have:
                wanted.setdefault(r["corpus"], []).append(r["question_id"])
        if not wanted:
            continue
        if cond == "bm25":
            got = bm25_evidence(icfg, wanted, questions, k)
        elif cond == "reference":
            got = reference_evidence(rag_dir, cfg["reference_variant"], wanted, k)
        else:
            raise ValueError(f"unknown condition {cond}")
        new = [{"condition": cond, "corpus": c, "question_id": q, "context_ids": ids} for (c, q), ids in got.items()]
        R.append_jsonl(path, new)
        for n in new:
            have[(cond, n["corpus"], n["question_id"])] = n["context_ids"]
    return have


# ---------------------------------------------------------------- report

def condition_rows(names_by_cond, models):
    """[(label, verifier key)] in report order: model change first, then evidence changes."""
    out = []
    for cond, names in names_by_cond.items():
        for m in models:
            if names.get(m):
                out.append((f"{CONDITION_TEXT[cond]}, verifier {m}", names[m], cond))
    return out


def markdown_report(meta, groups, order):
    lines = [f"# Generator-verifier independence: {meta['tag']}", "",
             f"Generator answers: results/rag/{meta['rag_tag']} ({meta['generator_model']}), fixed in every condition.",
             "Verifier models (same prompt, accept = P(Yes) > P(No)):", ""]
    for m, desc in meta["verifier_models"].items():
        lines.append(f"- **{m}**: {desc}")
    lines += ["", "Columns: false acceptance = verifier accepts an answer the generator got wrong; "
              "accept if followed poison = verifier accepts an answer that states the attack's false value; "
              "accept if correct = verifier keeps a right answer; overlap = share of the verifier's top-5 that is "
              "also in the generator's top-5; poison seen = share of answers whose verifier evidence contains a "
              "poisoned passage. Brackets = number of answers the rate is computed on.", ""]
    for g, block in groups.items():
        lines += [f"## {g} ({block['n']} answers, generator correct {V.fmt(block['generator_correct'])}"
                  + (f", ASR {V.fmt(block['asr'])}" if block['asr'] is not None else "") + ")", "",
                  "| Verifier condition | accept if correct | false acceptance | accept if followed poison "
                  "| overlap with generator | poison seen |", "|---|---|---|---|---|---|"]
        for label, key, _ in order:
            v = block["by_key"][key]
            lines.append(f"| {label} | {V.fmt(v['accept_if_correct'], v['n_correct'])} | "
                         f"{V.fmt(v['false_acceptance'], v['n_wrong'])} | "
                         f"{V.fmt(v['accept_if_followed_poison'], v['n_followed_poison'])} | "
                         f"{V.fmt(v['overlap'])} | {V.fmt(v['poison_seen'])} |")
        lines.append("")
    return "\n".join(lines)


def examples_markdown(rows, order, questions, passages, k):
    out = ["# Examples: poisoned answers, judged on each kind of evidence", ""]
    keys = [o[1] for o in order]
    sel = [r for r in rows if r["corpus"] != "clean" and r.get("attack_success")
           and any(r.get(f"accept_{x}") is False for x in keys) and any(r.get(f"accept_{x}") for x in keys)]
    for r in sorted(sel, key=lambda r: (r["corpus"], r["question_id"]))[:k]:
        out += [f"**{r['corpus']} / {r['question_id']}**: {questions[r['question_id']]['question']}",
                f"- gold: {r['gold_answer']} | generator (followed poison): {r['answer']}", ""]
        for label, key, cond in order:
            ids = r["context_ids"] if cond == "same_evidence" else r[f"evidence_{cond}"]
            out.append(f"- {label}: **{'accept' if r.get(f'accept_{key}') else 'reject'}** "
                       f"(P(Yes)={V.fmt(r.get(f'p_yes_{key}'))})")
            if key.startswith(order[0][1].split('@')[0]):  # list each evidence set once
                for pid in ids:
                    tag = " [POISON]" if pid in r["_poison_ids"] else ""
                    out.append(f"    - {passages[pid]['title']}{tag}: {G.truncate_words(passages[pid]['text'], 25)}")
        out.append("")
    if not sel:
        out.append("(no poisoned answer was accepted under one condition and rejected under another)")
    return "\n".join(out)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--tag", default=None, help="Output folder name (default: the rag tag).")
    ap.add_argument("--verifiers", nargs="*", help="Subset of verifier names (default: all).")
    ap.add_argument("--limit", type=int, default=0, help="Max answers per corpus (quick test).")
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--stub", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    cfg = R.load_yaml(args.config)
    vcfg = R.load_yaml(cfg["verifier_config"])
    rcfg = R.load_yaml(cfg["rag_config"])
    icfg = ic.load_config(rcfg["index_config"])
    clean_v, k = icfg["clean_variant"], rcfg["retrieval"]["top_k"]
    tag = args.tag or cfg["rag_tag"]
    out = os.path.join(cfg["results_root"], tag)
    os.makedirs(out, exist_ok=True)
    paths = {n: os.path.join(out, n) for n in ("run_meta.json", "evidence.jsonl", "verdicts.jsonl", "per_row.jsonl",
                                               "summary.json", "summary.md", "examples.md")}

    # ---- fixed inputs: generator answers, Task 2 same-evidence verdicts
    rag_dir = os.path.join(rcfg["results_root"], cfg["rag_tag"])
    rag_meta = ic.read_json(os.path.join(rag_dir, "run_meta.json"))
    rows = ic.read_jsonl(os.path.join(rag_dir, "per_question.jsonl"))
    if args.limit:
        keep, seen = [], {}
        for r in sorted(rows, key=lambda r: (r["corpus"], r["question_id"])):
            if seen.get(r["corpus"], 0) < args.limit:
                keep.append(r)
                seen[r["corpus"]] = seen.get(r["corpus"], 0) + 1
        rows = keep
    t2_dir = os.path.join(vcfg["results_root"], cfg["same_evidence_tag"])
    t2_meta = ic.read_json(os.path.join(t2_dir, "run_meta.json"))
    assert t2_meta["lock"]["evidence"]["rag_lock_hash"] == rag_meta["lock_hash"], \
        "Task 2 verifier results were made on other generator answers"
    assert t2_meta["lock"]["prompt"] == vcfg["prompt"] and t2_meta["lock"]["dtype"] == vcfg["dtype"], \
        "verifier prompt/dtype differ from the Task 2 run; the conditions would not be comparable"
    models = [v["name"] for v in vcfg["verifiers"] if not args.verifiers or v["name"] in args.verifiers]
    by_name = {v["name"]: v for v in vcfg["verifiers"]}
    revisions = {m: t2_meta["lock"]["verifiers"][m]["revision"] for m in by_name}  # the exact Task 2 commits

    variants = sorted({r["corpus"] for r in rows} - {clean_v})
    questions = st.load_questions(icfg)

    lock = {"conditions": cfg["conditions"], "reference_variant": cfg["reference_variant"], "top_k": k,
            "bm25": ic.bm25_lock(icfg), "verifiers": t2_meta["lock"]["verifiers"], "prompt": vcfg["prompt"],
            "dtype": vcfg["dtype"], "rag_lock_hash": rag_meta["lock_hash"], "task2_lock_hash": t2_meta["lock_hash"]}
    lock_h = R.lock_hash(lock)
    if os.path.exists(paths["run_meta.json"]):
        old = ic.read_json(paths["run_meta.json"])
        assert old["lock_hash"] == lock_h, f"{out} was made with other settings ({old['lock_hash']} vs {lock_h})"
    meta = {"tag": tag, "rag_tag": cfg["rag_tag"], "same_evidence_tag": cfg["same_evidence_tag"], "lock": lock,
            "lock_hash": lock_h, "generator_model": rag_meta["generator_model"],
            "verifier_models": {m: f"{by_name[m]['model_name']} @ {revisions[m]}" for m in by_name},
            "run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "limit": args.limit}
    ic.write_json(meta, paths["run_meta.json"])

    # ---- verifier evidence per condition
    evidence = build_evidence(cfg, icfg, rag_dir, rows, questions, k, paths["evidence.jsonl"])
    needed = {p for r in rows for p in r["context_ids"]} | {p for ids in evidence.values() for p in ids}
    passages = R.load_passages(icfg, variants, needed)
    log(f"{len(rows)} generator answers; evidence for {', '.join(cfg['conditions'])} ready")

    # ---- verification (one model load per verifier, all new conditions)
    if not args.score_only:
        for m in models:
            v = by_name[m]
            if args.stub:
                verifier = V.StubVerifier()
            else:
                log(f"loading verifier '{m}': {v['model_name']} @ {revisions[m]}")
                verifier = SplitOnOOM(V.HFVerifier(v["model_name"], revisions[m], vcfg["dtype"],
                                                   vcfg["prompt"]["yes_words"], vcfg["prompt"]["no_words"]))
            for cond in cfg["conditions"]:
                jobs = [(r["corpus"], r["question_id"],
                         V.verifier_messages(vcfg, rcfg, questions[r["question_id"]]["question"], r["answer"],
                                             [passages[p] for p in evidence[(cond, r["corpus"], r["question_id"])]]))
                        for r in rows]
                V.verify_all(vcfg, f"{m}@{cond}", verifier, jobs, paths["verdicts.jsonl"], lock_h, vcfg["batch_size"])
            del verifier
            V.free_gpu()

    # ---- merge: same-evidence verdicts from Task 2 + new conditions
    verdicts = {}
    for r in ic.read_jsonl(os.path.join(t2_dir, "verdicts.jsonl")):
        verdicts[(f"{r['verifier']}@same_evidence", r["corpus"], r["question_id"])] = r
    if os.path.exists(paths["verdicts.jsonl"]):
        for r in ic.read_jsonl(paths["verdicts.jsonl"]):
            assert r["lock_hash"] == lock_h, "verdicts.jsonl mixes settings"
            verdicts[(r["verifier"], r["corpus"], r["question_id"])] = r
    names_by_cond = {c: {m: f"{m}@{c}" for m in models} for c in ["same_evidence"] + cfg["conditions"]}
    order = [o for o in condition_rows(names_by_cond, models)
             if any(key[0] == o[1] for key in verdicts)]
    keys = [o[1] for o in order]

    attacks = {v: st.load_attack(icfg, v) for v in variants}
    poison_ids = {v: set().union(*(a["poison_ids"] for a in attacks[v].values())) for v in variants}
    clean_by_q = {r["question_id"]: r for r in rows if r["corpus"] == clean_v}
    merged = []
    for r in rows:
        m = {x: r[x] for x in ("corpus", "question_id", "split", "answer", "gold_answer", "context_ids", "correct",
                               "in_clean_sample")}
        pz = poison_ids.get(r["corpus"], set())
        m["_poison_ids"] = pz
        if r["corpus"] != clean_v:
            m["attack_success"] = r.get("attack_success")
            c = clean_by_q.get(r["question_id"])
            m["flipped"] = bool(c and c["correct"] and not r["correct"])
        gen = set(r["context_ids"])
        for cond in ["same_evidence"] + cfg["conditions"]:
            ids = r["context_ids"] if cond == "same_evidence" else evidence[(cond, r["corpus"], r["question_id"])]
            if cond != "same_evidence":
                m[f"evidence_{cond}"] = ids
            m[f"overlap_{cond}"] = len(gen & set(ids)) / max(len(ids), 1)
            m[f"poison_seen_{cond}"] = bool(pz & set(ids))
        for key in keys:
            v = verdicts.get((key, r["corpus"], r["question_id"]))
            m[f"accept_{key}"] = v["accept"] if v and v["p_yes"] is not None else None
            m[f"p_yes_{key}"] = v["p_yes"] if v else None
        m["corpus"] = "clean" if r["corpus"] == clean_v else r["corpus"]
        merged.append(m)
    R.write_jsonl(paths["per_row.jsonl"], [{x: y for x, y in m.items() if x != "_poison_ids"} for m in merged])

    groups = {"Clean corpus, baseline sample": [m for m in merged if m["corpus"] == "clean" and m["in_clean_sample"]]}
    for v in variants:
        groups[f"{v}: poisoned corpus"] = [m for m in merged if m["corpus"] == v]
    summary = {}
    for g, rs in groups.items():
        if not rs:
            continue
        base = V.group_summary(rs, [])
        block = {"n": base["n"], "generator_correct": base["generator_correct"], "asr": base["asr"], "by_key": {}}
        for label, key, cond in order:
            s = V.verifier_metrics(rs, key)
            s["overlap"] = sum(x[f"overlap_{cond}"] for x in rs) / len(rs)
            s["poison_seen"] = (sum(x[f"poison_seen_{cond}"] for x in rs) / len(rs)) if g != list(groups)[0] else None
            s["condition"], s["label"] = cond, label
            block["by_key"][key] = s
        summary[g] = block
    ic.write_json({"meta": meta, "groups": summary}, paths["summary.json"])
    with open(paths["summary.md"], "w", encoding="utf-8") as f:
        f.write(markdown_report(meta, summary, order))
    with open(paths["examples.md"], "w", encoding="utf-8") as f:
        f.write(examples_markdown(merged, order, questions, passages, vcfg["n_examples"]))
    missing = sum(1 for m in merged for key in keys if m[f"accept_{key}"] is None)
    log(f"done: {len(merged)} answers x {len(keys)} verifier conditions ({missing} without a score) -> {out}")
    print("\n" + open(paths["summary.md"], encoding="utf-8").read())


if __name__ == "__main__":
    main()
