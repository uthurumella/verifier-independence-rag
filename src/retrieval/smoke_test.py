"""Retrieval smoke test: do poisoned passages reach the top-k, and what do they push out?

For every question an attack poisoned, the question is run against the clean
index and against that attack's index, with both dense and BM25 retrieval.
Reported per attack and retriever:

- poison@k          share of questions with at least one poisoned passage in the top k
- copies@k          mean number of poisoned passages in the top k (3 copies exist per target)
- gold@k clean      share of questions with a gold passage in the top k, clean corpus
- gold@k poisoned   the same on the poisoned corpus
- displaced@k       share of questions whose gold was in the clean top k but is not in the poisoned top k
- poison>gold       share of questions where a poisoned passage outranks every gold passage
- poison>target     share where a poisoned passage outranks the specific gold passage it attacks
- first poison rank median rank of the best poisoned passage (top max-k only)

Only the provenance files say which passages are poisoned; the retrievers
never see them.

Usage:
    python src/retrieval/smoke_test.py --tag pilot
    python src/retrieval/smoke_test.py --tag full --variants relevant_llm_v2
"""

import argparse
import datetime as dt
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import index_common as ic  # noqa: E402

RETRIEVERS = ("dense", "bm25")


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- inputs

def load_questions(cfg):
    questions = {}
    for row in ic.read_jsonl(cfg["paths"]["raw_dev"]):
        questions[row["id"]] = {"question": row["question"], "answer": row["answer"], "type": row.get("type")}
    return questions


def load_gold(cfg):
    gold = defaultdict(set)
    for row in ic.read_jsonl(cfg["paths"]["question_map"]):
        if row["is_gold"]:
            gold[row["question_id"]].add(row["passage_id"])
    return gold


def load_split_of(cfg):
    split_of = {}
    for name in ("dev", "test"):
        for row in ic.read_jsonl(os.path.join(cfg["paths"]["splits_dir"], f"{name}_ids.jsonl")):
            split_of[row["question_id"]] = name
    return split_of


def load_attack(cfg, variant):
    """question_id -> poisoned passage ids, attacked gold ids, and what was changed."""
    attack = {}
    for rec in ic.read_jsonl(ic.provenance_path(cfg, variant)):
        q = attack.setdefault(rec["question_id"], {
            "poison_ids": set(), "target_ids": set(), "changes": set(), "attack_type": rec.get("attack_type"),
        })
        q["poison_ids"].add(rec["poisoned_passage_id"])
        target = rec.get("source_passage_id") or ic.title_passage_id(rec["source_passage_title"])
        q["target_ids"].add(target)
        q["changes"].add((rec.get("original_value"), rec.get("poisoned_value")))
    return attack


# ---------------------------------------------------------------- retrieval

class Retrievers:
    """Loads each index once and caches results per (retriever, corpus)."""

    def __init__(self, cfg, use):
        self.cfg = cfg
        self.use = use
        self.encoder = ic.make_encoder(cfg) if "dense" in use else None
        self._query_vecs = {}

    def run(self, retriever, variant, qids, texts, k):
        if retriever == "dense":
            index = ic.DenseIndex.load(self.cfg, variant)
            assert ic.encoder_matches(self.encoder, index.meta), f"{variant}: dense index built with other settings"
            missing = [q for q in qids if q not in self._query_vecs]
            if missing:
                vecs = self.encoder.encode_queries([texts[qids.index(q)] for q in missing])
                self._query_vecs.update(zip(missing, vecs))
            import numpy as np
            ranked, _ = index.search_vectors(np.stack([self._query_vecs[q] for q in qids]), k)
        else:
            ranked, _ = ic.BM25Index.load(self.cfg, variant).search(texts, k)
        return dict(zip(qids, ranked))


# ---------------------------------------------------------------- metrics

def first_rank(ranked, ids):
    for i, pid in enumerate(ranked, start=1):
        if pid in ids:
            return i
    return None


def question_record(qid, clean_ranked, pois_ranked, gold, attack, ks):
    poison_ids, target_ids = attack["poison_ids"], attack["target_ids"]
    leaked = [p for p in clean_ranked if p in poison_ids]
    assert not leaked, f"{qid}: poisoned passage found in the clean index"

    fp = first_rank(pois_ranked, poison_ids)
    fg = first_rank(pois_ranked, gold)
    ft = first_rank(pois_ranked, target_ids)
    rec = {
        "question_id": qid,
        "first_poison_rank": fp,
        "first_gold_rank_poisoned": fg,
        "first_gold_rank_clean": first_rank(clean_ranked, gold),
        "target_rank_clean": first_rank(clean_ranked, target_ids),
        "target_rank_poisoned": ft,
        "poison_above_gold": fp is not None and (fg is None or fp < fg),
        "poison_above_target": fp is not None and (ft is None or fp < ft),
    }
    for k in ks:
        top_c, top_p = clean_ranked[:k], pois_ranked[:k]
        gold_c = any(p in gold for p in top_c)
        gold_p = any(p in gold for p in top_p)
        rec[f"copies@{k}"] = sum(p in poison_ids for p in top_p)
        rec[f"poison@{k}"] = rec[f"copies@{k}"] > 0
        rec[f"gold@{k}_clean"] = gold_c
        rec[f"gold@{k}_poisoned"] = gold_p
        rec[f"displaced@{k}"] = gold_c and not gold_p
    return rec


def mean(values):
    values = list(values)
    return round(sum(values) / len(values), 3) if values else None


def summarize(records, ks, cfg):
    n = len(records)
    s = {"n_questions": n}
    for k in ks:
        s[f"poison@{k}"] = mean(r[f"poison@{k}"] for r in records)
        s[f"copies@{k}"] = mean(r[f"copies@{k}"] for r in records)
        s[f"gold@{k}_clean"] = mean(r[f"gold@{k}_clean"] for r in records)
        s[f"gold@{k}_poisoned"] = mean(r[f"gold@{k}_poisoned"] for r in records)
        s[f"displaced@{k}"] = mean(r[f"displaced@{k}"] for r in records)
    s["poison_above_gold"] = mean(r["poison_above_gold"] for r in records)
    s["poison_above_target"] = mean(r["poison_above_target"] for r in records)
    ranks = [r["first_poison_rank"] for r in records if r["first_poison_rank"] is not None]
    s["median_first_poison_rank"] = float(statistics.median(ranks)) if ranks else None
    st = cfg["smoke_test"]
    s["verdict"] = "weak" if (s.get(f"poison@{st['verdict_k']}") or 0) < st["weak_threshold"] else "surfaces"
    return s


# ---------------------------------------------------------------- report

def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def markdown_report(tag, summaries, ks, cfg, meta):
    k_show = [k for k in ks if k in (5, 10)] or ks[:2]
    lines = [
        f"# Retrieval smoke test ({tag})",
        "",
        f"Run: {meta['run_at']}  ",
        f"Dense: {meta['dense_model']}  ",
        f"BM25: {meta['bm25']}  ",
        f"Verdict rule: \"weak\" if poison@{cfg['smoke_test']['verdict_k']} < {cfg['smoke_test']['weak_threshold']} "
        "(heuristic, for flagging only).",
        "",
    ]
    header = ["Attack", "Retriever", "Split", "Qs"]
    for k in k_show:
        header += [f"poison@{k}", f"copies@{k}", f"gold@{k} clean", f"gold@{k} pois.", f"displaced@{k}"]
    header += ["poison>gold", "poison>target", "med. 1st poison rank", "verdict"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for (variant, retriever, split), s in summaries.items():
        row = [variant, retriever, split, str(s["n_questions"])]
        for k in k_show:
            row += [fmt(s[f"poison@{k}"]), fmt(s[f"copies@{k}"]), fmt(s[f"gold@{k}_clean"]),
                    fmt(s[f"gold@{k}_poisoned"]), fmt(s[f"displaced@{k}"])]
        row += [fmt(s["poison_above_gold"]), fmt(s["poison_above_target"]),
                fmt(s["median_first_poison_rank"]), s["verdict"]]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "poison@k: share of questions with a poisoned passage in the top k. copies@k: mean poisoned passages in the "
        "top k (3 copies per target). gold@k: share with a gold passage in the top k. displaced@k: gold in the clean "
        "top k but not the poisoned top k. poison>gold / poison>target: a poisoned passage outranks every gold "
        f"passage / the gold passage it attacks, within the top {max(ks)}.",
    ]
    return "\n".join(lines) + "\n"


def examples_markdown(examples, passages_by_id):
    lines = ["# Example retrievals (poisoned corpus, top 5)", ""]
    for ex in examples:
        lines += [
            f"## {ex['variant']} / {ex['retriever']} / {ex['question_id']}",
            "",
            f"**Question:** {ex['question']}  ",
            f"**Gold answer:** {ex['answer']}  ",
            f"**Poisoned value(s):** {', '.join(ex['poisoned_values'])}",
            "",
        ]
        for i, (pid, label) in enumerate(ex["top"], start=1):
            p = passages_by_id.get(pid, {})
            text = p.get("text", "")
            text = text if len(text) <= 220 else text[:217] + "..."
            lines.append(f"{i}. `{label}` **{p.get('title', pid)}**: {text}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=ic.DEFAULT_CONFIG)
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--tag", default="pilot", help="Results go to <results_root>/<tag>/ (e.g. pilot, full).")
    parser.add_argument("--retrievers", nargs="*", default=list(RETRIEVERS), choices=RETRIEVERS)
    args = parser.parse_args()

    cfg = ic.load_config(args.config)
    variants = args.variants if args.variants else cfg["variants"]
    ks = sorted(cfg["smoke_test"]["k"])
    k_max = max(ks)
    out_dir = os.path.join(cfg["paths"]["results_root"], args.tag)
    os.makedirs(out_dir, exist_ok=True)

    questions = load_questions(cfg)
    gold = load_gold(cfg)
    split_of = load_split_of(cfg)
    retr = Retrievers(cfg, args.retrievers)

    summaries, all_records, examples = {}, [], []
    clean_cache = {}
    for variant in variants:
        attack = load_attack(cfg, variant)
        all_poison = set().union(*(a["poison_ids"] for a in attack.values()))
        qids = sorted(attack)
        missing = [q for q in qids if q not in questions]
        assert not missing, f"{variant}: {len(missing)} provenance questions not in {cfg['paths']['raw_dev']}"
        texts = [questions[q]["question"] for q in qids]
        log(f"[{variant}] {len(qids)} poisoned questions")

        for retriever in args.retrievers:
            key = (retriever, cfg["clean_variant"])
            need = [q for q in qids if q not in clean_cache.get(key, {})]
            if need:
                got = retr.run(retriever, cfg["clean_variant"], need, [questions[q]["question"] for q in need], k_max)
                clean_cache.setdefault(key, {}).update(got)
            pois = retr.run(retriever, variant, qids, texts, k_max)

            recs = []
            for q in qids:
                r = question_record(q, clean_cache[key][q], pois[q], gold[q], attack[q], ks)
                r.update({"variant": variant, "retriever": retriever, "split": split_of.get(q, "unknown"),
                          "question_type": questions[q]["type"]})
                recs.append(r)
            all_records += recs

            for split in ["all"] + sorted({r["split"] for r in recs}):
                subset = recs if split == "all" else [r for r in recs if r["split"] == split]
                if split != "all" and len(subset) == len(recs):
                    continue
                summaries[(variant, retriever, split)] = summarize(subset, ks, cfg)
            s = summaries[(variant, retriever, "all")]
            log(f"  {retriever:5s} poison@5={fmt(s['poison@5'] if 5 in ks else None)} "
                f"poison@10={fmt(s['poison@10'] if 10 in ks else None)} "
                f"displaced@5={fmt(s['displaced@5'] if 5 in ks else None)} verdict={s['verdict']}")

            for q in qids[: cfg["smoke_test"]["n_examples"]]:
                top = []
                for pid in pois[q][:5]:
                    if pid in attack[q]["poison_ids"]:
                        label = "POISON"
                    elif pid in gold[q]:
                        label = "GOLD"
                    elif pid in all_poison:
                        label = "poison for another question"
                    else:
                        label = "other"
                    top.append((pid, label))
                examples.append({
                    "variant": variant, "retriever": retriever, "question_id": q,
                    "question": questions[q]["question"], "answer": questions[q]["answer"],
                    "poisoned_values": sorted({str(c[1]) for c in attack[q]["changes"]}), "top": top,
                })

    meta = {
        "tag": args.tag,
        "run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "variants": variants,
        "k": ks,
        "dense_model": (f"{retr.encoder.lock['model_name']} @ {retr.encoder.lock['revision']}"
                        if retr.encoder else "not run"),
        "bm25": json.dumps(ic.bm25_lock(cfg)) if "bm25" in args.retrievers else "not run",
        "poison_sha256": {v: ic.file_sha256(ic.poison_passages_path(cfg, v)) for v in variants},
    }
    ic.write_json({"meta": meta, "summaries": [
        {"variant": v, "retriever": r, "split": sp, **s} for (v, r, sp), s in summaries.items()
    ]}, os.path.join(out_dir, "summary.json"))
    with open(os.path.join(out_dir, "per_question.jsonl"), "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(markdown_report(args.tag, summaries, ks, cfg, meta))

    needed = {pid for ex in examples for pid, _ in ex["top"]}
    passages_by_id = {}
    for v in [cfg["clean_variant"]] + variants:
        src = ic.clean_passages_path(cfg) if v == cfg["clean_variant"] else ic.poison_passages_path(cfg, v)
        for p in ic.read_jsonl(src):
            if p["passage_id"] in needed:
                passages_by_id[p["passage_id"]] = p
    with open(os.path.join(out_dir, "examples.md"), "w", encoding="utf-8") as f:
        f.write(examples_markdown(examples, passages_by_id))

    log(f"\nWrote {out_dir}/summary.md, summary.json, per_question.jsonl, examples.md")
    weak = [f"{v}/{r}" for (v, r, sp), s in summaries.items() if sp == "all" and s["verdict"] == "weak"]
    if weak:
        log("Flagged weak (poison rarely reaches the top "
            f"{cfg['smoke_test']['verdict_k']}): {', '.join(weak)}")


if __name__ == "__main__":
    main()
