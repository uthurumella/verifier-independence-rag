"""Task 2: standard RAG baseline on the clean corpus and each poisoned corpus.

Pipeline per question: dense retrieval (e5-base-v2 + FAISS index from Task 1)
-> top-k passages -> instruction-tuned LLM -> short answer -> scoring.

Question sets:
- every question an attack poisoned, answered on the clean corpus AND on that
  attack's corpus (paired, same settings)
- a fixed random sample of dev questions answered on the clean corpus only,
  for the overall clean baseline

Reported per attack and split:
- EM / F1 / correct (F1>=0.5) on clean and poisoned corpora
- retrieval recall: gold recall@k, any gold@k, both gold@k
- poison in context: share of questions with a poisoned passage in the top-k
- attack-success rate (ASR): answer states the false value, not the true one
- ASR given poison in context, flipped (correct on clean -> attacked on poisoned)

Provenance is only read here, for scoring; retriever and generator never see it.

Usage:
    python src/rag/run_rag.py --tag pilot
    python src/rag/run_rag.py --tag pilot --variants relevant_llm_v1 --limit 20
    python src/rag/run_rag.py --tag pilot --score_only       # rebuild reports from predictions.jsonl

Outputs in results/rag/<tag>/:
    run_meta.json        locked settings, model commits, input hashes
    retrieval.jsonl      top-k passage ids per (corpus, question)
    predictions.jsonl    append-only checkpoint of generated answers (resumable)
    per_question.jsonl   one scored row per (corpus, question)
    summary.json / summary.md / examples.md
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "retrieval"))
import index_common as ic  # noqa: E402
import smoke_test as st  # noqa: E402  (reuse its loaders)
import metrics as M  # noqa: E402
import generator as G  # noqa: E402

DEFAULT_RAG_CONFIG = "configs/rag_config.yaml"


def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- run lock

def run_lock(rcfg, icfg, gen_revision, dense_revision):
    g = {k: v for k, v in rcfg["generator"].items() if k != "batch_size"}
    g["revision"] = gen_revision
    return {
        "top_k": rcfg["retrieval"]["top_k"],
        "generator": g,
        "prompt": rcfg["prompt"],
        "dense": ic.dense_lock(icfg, dense_revision),
    }


def lock_hash(lock):
    return hashlib.sha256(json.dumps(lock, sort_keys=True).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- jobs

def select_questions(rcfg, icfg, variants, questions, split_of, attacks, limit):
    """Returns {corpus: [qids]} and the clean-sample qid list."""
    q = rcfg["questions"]
    jobs = {}
    attacked_union = set()
    for v in variants:
        qids = sorted(attacks[v])
        if limit:
            qids = qids[:limit]
        jobs[v] = qids
        attacked_union.update(qids)

    pool = sorted(qid for qid, s in split_of.items() if q["clean_sample_split"] in ("all", s) and qid in questions)
    rng = random.Random(q["seed"])
    rng.shuffle(pool)
    n = q["clean_sample_size"] if not limit else min(q["clean_sample_size"], limit)
    sample = pool[:n]
    jobs[icfg["clean_variant"]] = sorted(attacked_union | set(sample))
    return jobs, sample


# ---------------------------------------------------------------- retrieval

def retrieve_all(icfg, jobs, questions, k_max, out_path):
    """Dense top-k_max for every (corpus, qid). Cached in retrieval.jsonl."""
    done = {}
    if os.path.exists(out_path):
        for r in ic.read_jsonl(out_path):
            done[(r["corpus"], r["question_id"])] = r["ranked"]
    todo = {c: [q for q in qids if (c, q) not in done] for c, qids in jobs.items()}
    if not any(todo.values()):
        log("retrieval: all cached")
        return done

    import numpy as np
    encoder = ic.make_encoder(icfg)
    need_q = sorted({q for qids in todo.values() for q in qids})
    log(f"retrieval: encoding {len(need_q)} queries")
    vecs = dict(zip(need_q, encoder.encode_queries([questions[q]["question"] for q in need_q])))
    for corpus, qids in todo.items():
        if not qids:
            continue
        index = ic.DenseIndex.load(icfg, corpus)
        assert ic.encoder_matches(encoder, index.meta), f"{corpus}: index built with other encoder settings"
        ranked, _ = index.search_vectors(np.stack([vecs[q] for q in qids]), k_max)
        rows = [{"corpus": corpus, "question_id": q, "ranked": r} for q, r in zip(qids, ranked)]
        append_jsonl(out_path, rows)
        for r in rows:
            done[(corpus, r["question_id"])] = r["ranked"]
        log(f"retrieval: {corpus} {len(qids)} questions")
        del index
    revision = encoder.lock["revision"]
    del encoder
    _free_gpu()
    return done, revision


def _free_gpu():
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_passages(icfg, variants, needed):
    by_id = {}
    for v in [icfg["clean_variant"]] + list(variants):
        src = ic.clean_passages_path(icfg) if v == icfg["clean_variant"] else ic.poison_passages_path(icfg, v)
        for p in ic.read_jsonl(src):
            if p["passage_id"] in needed:
                by_id[p["passage_id"]] = p
    missing = needed - set(by_id)
    assert not missing, f"{len(missing)} retrieved passage ids not found in any corpus file"
    return by_id


# ---------------------------------------------------------------- generation

class StubGenerator:
    """For pipeline tests without a GPU: answers with the first sentence of the top passage."""
    revision = "stub"

    def generate(self, list_of_messages):
        out = []
        for m in list_of_messages:
            lines = m[1]["content"].split("\n")
            text = lines[2] if len(lines) > 2 else ""
            ans = text.split(". ")[0]
            if " known for " in ans:
                ans = ans.split(" known for ", 1)[1]
            out.append({"raw": ans, "answer": G.clean_answer(ans), "prompt_tokens": len(m[1]["content"].split())})
        return out


def generate_all(rcfg, gen, jobs_list, pred_path, prompt_h, lock_h):
    """jobs_list: [(corpus, qid, messages)]. Appends to predictions.jsonl, skipping finished ones."""
    done = set()
    if os.path.exists(pred_path):
        for r in ic.read_jsonl(pred_path):
            assert r["lock_hash"] == lock_h, "predictions.jsonl was made with other settings; use a new --tag"
            done.add((r["corpus"], r["question_id"]))
    todo = [j for j in jobs_list if (j[0], j[1]) not in done]
    log(f"generation: {len(done)} cached, {len(todo)} to run")
    if not todo:
        return
    todo.sort(key=lambda j: len(j[2][1]["content"]))  # similar lengths per batch, less padding
    bs = rcfg["generator"]["batch_size"]
    t0 = time.time()
    for i in range(0, len(todo), bs):
        batch = todo[i:i + bs]
        outs = gen.generate([j[2] for j in batch])
        rows = [{
            "corpus": c, "question_id": q, "answer": o["answer"], "raw": o["raw"],
            "prompt_tokens": o["prompt_tokens"], "prompt_hash": prompt_h, "lock_hash": lock_h,
        } for (c, q, _), o in zip(batch, outs)]
        append_jsonl(pred_path, rows)
        n = i + len(batch)
        if (i // bs) % 10 == 0 or n == len(todo):
            rate = n / max(time.time() - t0, 1e-6)
            log(f"generation: {n}/{len(todo)}  ({rate:.1f} q/s, ~{(len(todo) - n) / max(rate, 1e-6) / 60:.1f} min left)")


# ---------------------------------------------------------------- scoring

def retrieval_fields(ranked, gold, poison_ids, top_k, ks):
    rec = {}
    for k in sorted(set(ks) | {top_k}):
        top = ranked[:k]
        hit = sum(p in gold for p in top)
        rec[f"gold_recall@{k}"] = hit / len(gold) if gold else None
        rec[f"any_gold@{k}"] = hit > 0
        rec[f"all_gold@{k}"] = hit == len(gold) and bool(gold)
    ctx = ranked[:top_k]
    rec["poison_in_context"] = sum(p in poison_ids for p in ctx)
    rec["poison_ranks"] = [i for i, p in enumerate(ctx, start=1) if p in poison_ids]
    return rec


def mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize_pairs(rows_clean, rows_pois, top_k):
    """Paired summary for one attack: same questions, clean vs poisoned corpus."""
    n = len(rows_pois)
    pc = {r["question_id"]: r for r in rows_clean}
    s = {"n_questions": n}
    for name, rows in (("clean", rows_clean), ("poisoned", rows_pois)):
        s[f"em_{name}"] = mean(r["em"] for r in rows)
        s[f"f1_{name}"] = mean(r["f1"] for r in rows)
        s[f"correct_{name}"] = mean(r["correct"] for r in rows)
        s[f"gold_recall@{top_k}_{name}"] = mean(r[f"gold_recall@{top_k}"] for r in rows)
        s[f"any_gold@{top_k}_{name}"] = mean(r[f"any_gold@{top_k}"] for r in rows)
        s[f"asr_{name}"] = mean(r["attack_success"] for r in rows)
    s["asr_em_poisoned"] = mean(r["attack_success_em"] for r in rows_pois)
    s["poison_in_context"] = mean(r["poison_in_context"] > 0 for r in rows_pois)
    s["mean_poison_in_context"] = mean(r["poison_in_context"] for r in rows_pois)
    with_p = [r for r in rows_pois if r["poison_in_context"] > 0]
    s["n_poison_in_context"] = len(with_p)
    s["asr_given_poison_in_context"] = mean(r["attack_success"] for r in with_p)
    s["correct_given_poison_in_context"] = mean(r["correct"] for r in with_p)
    was_right = [r for r in rows_pois if pc[r["question_id"]]["correct"]]
    s["n_correct_clean"] = len(was_right)
    s["flipped_to_poison"] = mean(r["attack_success"] for r in was_right)
    s["broken_any"] = mean(not r["correct"] for r in was_right)
    return s


def summarize_clean(rows, top_k, ks):
    s = {"n_questions": len(rows), "em": mean(r["em"] for r in rows), "f1": mean(r["f1"] for r in rows),
         "correct": mean(r["correct"] for r in rows), "gold_in_pred": mean(r["gold_in_pred"] for r in rows),
         "mean_prompt_tokens": mean(r["prompt_tokens"] for r in rows)}
    for k in sorted(set(ks) | {top_k}):
        s[f"gold_recall@{k}"] = mean(r[f"gold_recall@{k}"] for r in rows)
        s[f"all_gold@{k}"] = mean(r[f"all_gold@{k}"] for r in rows)
    return s


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def markdown_report(tag, meta, clean_summary, attack_summaries, top_k, clean_by_split=None):
    L = [f"# Standard RAG baseline ({tag})", "",
         f"Run: {meta['run_at']}  ",
         f"Retriever: {meta['dense_model']}, top-{top_k} passages  ",
         f"Generator: {meta['generator_model']}, greedy, max {meta['max_new_tokens']} new tokens  ",
         f"Prompt hash: {meta['prompt_hash']}, run lock: {meta['lock_hash']}", "",
         "## Clean baseline (clean corpus)", ""]
    ks = [k for k in (1, top_k, 10) if f"gold_recall@{k}" in clean_summary]
    hdr = ["Split", "Qs", "EM", "F1", "correct (F1>=0.5)", "gold in answer"] + [f"gold recall@{k}" for k in ks] + \
          [f"both gold@{k}" for k in ks] + ["prompt tokens"]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for name, cs in [("all", clean_summary)] + sorted((clean_by_split or {}).items()):
        row = [name, cs["n_questions"], cs["em"], cs["f1"], cs["correct"], cs["gold_in_pred"]] + \
              [cs[f"gold_recall@{k}"] for k in ks] + [cs[f"all_gold@{k}"] for k in ks] + [cs["mean_prompt_tokens"]]
        L.append("| " + " | ".join(fmt(x) for x in row) + " |")
    L.append("")
    if not attack_summaries:
        return "\n".join(L)

    L += ["## Attacked questions: clean corpus vs poisoned corpus (same questions)", ""]
    hdr = ["Attack", "Split", "Qs", "EM clean→pois.", "F1 clean→pois.", "correct clean→pois.",
           f"gold recall@{top_k} clean→pois.", "poison in top-k", "ASR", "ASR (EM)", "ASR given poison in top-k",
           "flipped (correct→false)", "ASR on clean corpus"]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for (v, split), s in attack_summaries.items():
        L.append("| " + " | ".join([
            v, split, str(s["n_questions"]),
            f"{fmt(s['em_clean'])}→{fmt(s['em_poisoned'])}",
            f"{fmt(s['f1_clean'])}→{fmt(s['f1_poisoned'])}",
            f"{fmt(s['correct_clean'])}→{fmt(s['correct_poisoned'])}",
            f"{fmt(s[f'gold_recall@{top_k}_clean'])}→{fmt(s[f'gold_recall@{top_k}_poisoned'])}",
            fmt(s["poison_in_context"]), fmt(s["asr_poisoned"]), fmt(s["asr_em_poisoned"]),
            fmt(s["asr_given_poison_in_context"]), fmt(s["flipped_to_poison"]), fmt(s["asr_clean"]),
        ]) + " |")
    L += ["",
          "ASR: share of attacked questions whose answer states the false value from provenance and not the true "
          "one. ASR (EM): answer exactly equals the poisoned answer. ASR given poison in top-k: ASR among questions "
          "where a poisoned passage reached the generator. flipped: among questions answered correctly on the clean "
          "corpus, share that give the false answer on the poisoned corpus. ASR on clean corpus is a sanity check "
          "(should be near 0). correct = F1 >= 0.5 against the gold answer.", ""]
    return "\n".join(L)


def examples_markdown(rows_by_key, passages, attacks, questions, n=3):
    L = ["# Example answers (clean vs poisoned corpus)", ""]
    for v, att in attacks.items():
        flips = [q for q in sorted(att) if (v, q) in rows_by_key and rows_by_key[(v, q)]["attack_success"]]
        keep = [q for q in sorted(att) if (v, q) in rows_by_key and not rows_by_key[(v, q)]["attack_success"]]
        for q in flips[:n] + keep[:1]:
            rp = rows_by_key[(v, q)]
            rc = rows_by_key.get(("clean", q))
            L += [f"## {v} / {q} ({'attack succeeded' if rp['attack_success'] else 'attack failed'})", "",
                  f"**Question:** {questions[q]['question']}  ",
                  f"**Gold:** {questions[q]['answer']}  ·  **False value:** {rp['poisoned_answer']}  ",
                  f"**Answer, clean corpus:** {rc['answer'] if rc else '-'}  ",
                  f"**Answer, poisoned corpus:** {rp['answer']}", "", "Poisoned-corpus context:"]
            for i, pid in enumerate(rp["context_ids"], start=1):
                p = passages.get(pid, {})
                tag = "POISON" if pid in att[q]["poison_ids"] else ("GOLD" if pid in rp["_gold"] else "other")
                text = p.get("text", "")
                L.append(f"{i}. `{tag}` **{p.get('title', pid)}**: {text[:200]}{'...' if len(text) > 200 else ''}")
            L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=DEFAULT_RAG_CONFIG)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--limit", type=int, default=0, help="Max attacked questions per attack (quick test).")
    ap.add_argument("--score_only", action="store_true", help="Only rebuild reports from saved predictions.")
    ap.add_argument("--stub_generator", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    rcfg = load_yaml(args.config)
    icfg = ic.load_config(rcfg["index_config"])
    variants = args.variants if args.variants else icfg["variants"]
    if variants == ["none"]:
        variants = []  # clean corpus only
    top_k = rcfg["retrieval"]["top_k"]
    ks = rcfg["retrieval"]["report_k"]
    k_max = max(ks + [top_k])
    clean_v = icfg["clean_variant"]

    out = os.path.join(rcfg["results_root"], args.tag)
    os.makedirs(out, exist_ok=True)
    paths = {n: os.path.join(out, n) for n in
             ("run_meta.json", "retrieval.jsonl", "predictions.jsonl", "per_question.jsonl",
              "summary.json", "summary.md", "examples.md")}

    questions = st.load_questions(icfg)
    gold = st.load_gold(icfg)
    split_of = st.load_split_of(icfg)
    attacks = {v: st.load_attack(icfg, v) for v in variants}
    for v in variants:
        bad = [q for q in attacks[v] if len(attacks[v][q]["changes"]) != 1]
        assert not bad, f"{v}: {len(bad)} questions with more than one (true, false) value pair"
    jobs, clean_sample = select_questions(rcfg, icfg, variants, questions, split_of, attacks, args.limit)
    log(f"questions: clean corpus {len(jobs[clean_v])} (sample {len(clean_sample)}), "
        + ", ".join(f"{v} {len(jobs[v])}" for v in variants))

    # ---- locked settings
    meta_path = paths["run_meta.json"]
    if os.path.exists(meta_path):
        meta = ic.read_json(meta_path)
        gen_rev = meta["lock"]["generator"]["revision"]
        dense_rev = meta["lock"]["dense"]["revision"]
    else:
        gen_rev = "stub" if args.stub_generator else G.resolve_model_revision(
            rcfg["generator"]["model_name"], rcfg["generator"]["revision"])
        dense_rev = ic.resolve_revision(icfg)
    lock = run_lock(rcfg, icfg, gen_rev, dense_rev)
    lock_h = lock_hash(lock)
    if os.path.exists(meta_path):
        assert meta["lock_hash"] == lock_h, (
            f"{out} was started with other locked settings ({meta['lock_hash']} vs {lock_h}); use a new --tag")
    meta = {
        "tag": args.tag, "lock": lock, "lock_hash": lock_h, "prompt_hash": G.prompt_hash(rcfg),
        "run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dense_model": f"{lock['dense']['model_name']} @ {dense_rev}",
        "generator_model": f"{rcfg['generator']['model_name']} @ {gen_rev}",
        "max_new_tokens": rcfg["generator"]["max_new_tokens"],
        "variants": variants, "limit": args.limit,
        "n_questions": {c: len(q) for c, q in jobs.items()}, "clean_sample_size": len(clean_sample),
        "inputs_sha256": {"clean_passages": ic.file_sha256(ic.clean_passages_path(icfg)),
                          **{v: ic.file_sha256(ic.poison_passages_path(icfg, v)) for v in variants}},
    }
    ic.write_json(meta, meta_path)

    # ---- retrieval
    if args.score_only:
        ranked = {(r["corpus"], r["question_id"]): r["ranked"] for r in ic.read_jsonl(paths["retrieval.jsonl"])}
    else:
        got = retrieve_all(icfg, jobs, questions, k_max, paths["retrieval.jsonl"])
        ranked = got[0] if isinstance(got, tuple) else got
    needed = {p for (c, q), r in ranked.items() if c in jobs for p in r[:top_k]}
    passages = load_passages(icfg, variants, needed)

    # ---- generation
    pred_path = paths["predictions.jsonl"]
    if not args.score_only:
        jobs_list = []
        for corpus, qids in jobs.items():
            for q in qids:
                ctx = [passages[p] for p in ranked[(corpus, q)][:top_k]]
                jobs_list.append((corpus, q, G.build_messages(rcfg, questions[q]["question"], ctx)))
        if args.stub_generator:
            gen = StubGenerator()
        else:
            log(f"loading generator {rcfg['generator']['model_name']} @ {gen_rev}")
            gen = G.HFGenerator(rcfg, gen_rev)
        generate_all(rcfg, gen, jobs_list, pred_path, meta["prompt_hash"], lock_h)
        del gen
        _free_gpu()

    preds = {}
    for r in ic.read_jsonl(pred_path):
        assert r["lock_hash"] == lock_h, "predictions.jsonl mixes settings"
        preds[(r["corpus"], r["question_id"])] = r

    # ---- scoring
    correct_f1 = rcfg["scoring"]["correct_f1"]
    all_poison = {v: set().union(*(a["poison_ids"] for a in attacks[v].values())) for v in variants}
    rows, rows_by_key = [], {}
    clean_sample_set = set(clean_sample)
    for corpus, qids in jobs.items():
        for q in qids:
            p = preds.get((corpus, q))
            if p is None:
                continue
            change = None
            for v in variants:  # the attack this question is scored against
                if q in attacks[v] and (corpus == v or corpus == clean_v):
                    change = next(iter(attacks[v][q]["changes"]))
                    break
            r = {"corpus": corpus, "question_id": q, "split": split_of.get(q, "unknown"),
                 "question_type": questions[q]["type"], "answer": p["answer"], "gold_answer": questions[q]["answer"],
                 "prompt_tokens": p["prompt_tokens"], "context_ids": ranked[(corpus, q)][:top_k],
                 "in_clean_sample": q in clean_sample_set}
            poison_ids = all_poison.get(corpus, set())
            r.update(retrieval_fields(ranked[(corpus, q)], gold[q], poison_ids, top_k, ks))
            r.update(M.score_answer(p["answer"], questions[q]["answer"], change, correct_f1))
            rows.append(r)
            rows_by_key[(corpus, q)] = dict(r, _gold=gold[q])
    write_jsonl(paths["per_question.jsonl"], rows)

    clean_rows = [r for r in rows if r["corpus"] == clean_v and r["question_id"] in clean_sample_set]
    clean_summary = summarize_clean(clean_rows, top_k, ks)
    clean_by_split = {sp: summarize_clean([r for r in clean_rows if r["split"] == sp], top_k, ks)
                      for sp in ("dev", "test") if any(r["split"] == sp for r in clean_rows)}

    attack_summaries = {}
    for v in variants:
        pois = [rows_by_key[(v, q)] for q in jobs[v] if (v, q) in rows_by_key]
        # clean-corpus rows scored against this attack's false value
        clean = []
        for q in jobs[v]:
            if (clean_v, q) not in rows_by_key:
                continue
            c = dict(rows_by_key[(clean_v, q)])
            c.update(M.score_answer(c["answer"], questions[q]["answer"], next(iter(attacks[v][q]["changes"])),
                                    correct_f1))
            clean.append(c)
        ok = {r["question_id"] for r in pois} & {r["question_id"] for r in clean}
        pois = [r for r in pois if r["question_id"] in ok]
        clean = [r for r in clean if r["question_id"] in ok]
        for split in ["all", "dev", "test"]:
            sp = pois if split == "all" else [r for r in pois if r["split"] == split]
            sc = clean if split == "all" else [r for r in clean if r["split"] == split]
            if not sp or (split != "all" and len(sp) == len(pois)):
                continue
            attack_summaries[(v, split)] = summarize_pairs(sc, sp, top_k)

    ic.write_json({"meta": meta, "clean_baseline": clean_summary, "clean_baseline_by_split": clean_by_split, "attacks": [
        {"variant": v, "split": sp, **s} for (v, sp), s in attack_summaries.items()]}, paths["summary.json"])
    with open(paths["summary.md"], "w", encoding="utf-8") as f:
        f.write(markdown_report(args.tag, meta, clean_summary, attack_summaries, top_k, clean_by_split))
    rows_by_key_named = {("clean" if c == clean_v else c, q): r for (c, q), r in rows_by_key.items()}
    with open(paths["examples.md"], "w", encoding="utf-8") as f:
        f.write(examples_markdown(rows_by_key_named, passages, {v: attacks[v] for v in variants}, questions))

    missing = sum(len(q) for q in jobs.values()) - len(rows)
    log(f"done: {len(rows)} scored answers ({missing} missing) -> {out}/summary.md")
    print("\n" + open(paths["summary.md"], encoding="utf-8").read())


if __name__ == "__main__":
    main()
