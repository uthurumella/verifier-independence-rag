"""Builds dense (FAISS) and BM25 indexes for the clean corpus and each poisoned corpus.

The same command is run for the pilot data now and again after full-scale
poisoning. Work is skipped when its inputs have not changed:

- The clean corpus is embedded once. Its index is reused as long as
  passages.jsonl and the locked settings in configs/index_config.yaml are
  unchanged.
- For a poisoned corpus only its own poisoned passages are embedded. They are
  re-embedded only when poisoned_passages.jsonl changes, and the clean vectors
  are copied from the clean index rather than recomputed.
- BM25 is rebuilt for any corpus whose passages changed, since adding passages
  changes the term statistics of the whole corpus.

If a locked setting differs from what an existing index was built with, the
script stops instead of silently mixing two embedding spaces. Pass --rebuild to
start over on purpose.

Usage:
    python src/retrieval/build_indexes.py                  # clean + all variants in the config
    python src/retrieval/build_indexes.py --variants relevant_llm_v1
    python src/retrieval/build_indexes.py --skip_dense     # BM25 only, no GPU needed
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import index_common as ic  # noqa: E402


def log(msg):
    print(msg, flush=True)


def check_lock(meta_path, lock, what):
    """True if an index exists and matches; False if absent; error if it conflicts."""
    if not os.path.exists(meta_path):
        return None
    meta = ic.read_json(meta_path)
    if meta["lock"] != lock:
        diff = {k: (meta["lock"].get(k), lock.get(k)) for k in lock if meta["lock"].get(k) != lock.get(k)}
        raise SystemExit(
            f"{what} was built with different locked settings (existing, requested): {diff}\n"
            "Restore the old settings in configs/index_config.yaml, or pass --rebuild to rebuild everything."
        )
    return meta


def write_ids(cfg, variant, ids):
    path = os.path.join(ic.index_dir(cfg, variant), "passage_ids.json")
    if os.path.exists(path) and ic.read_json(path) != ids:
        log(f"  [{variant}] passage order changed; rewriting passage_ids.json")
    ic.write_json(ids, path)


# ---------------------------------------------------------------- dense

def build_dense_clean(cfg, encoder):
    import faiss

    variant = cfg["clean_variant"]
    out = os.path.join(ic.index_dir(cfg, variant), "dense")
    meta_path = os.path.join(out, "meta.json")
    clean_hash = ic.file_sha256(ic.clean_passages_path(cfg))

    meta = check_lock(meta_path, encoder.lock, f"Clean dense index ({out})")
    if meta and meta["clean_sha256"] == clean_hash and os.path.exists(os.path.join(out, "index.faiss")):
        log(f"  [{variant}] dense: unchanged, reusing")
        return
    if meta:
        log(f"  [{variant}] dense: clean passages changed, re-embedding")

    passages = ic.load_corpus(cfg, variant)
    log(f"  [{variant}] dense: embedding {len(passages):,} passages")
    t = time.time()
    emb = encoder.encode_passages(passages)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    os.makedirs(out, exist_ok=True)
    faiss.write_index(index, os.path.join(out, "index.faiss"))
    write_ids(cfg, variant, [p["passage_id"] for p in passages])
    ic.write_json({
        "variant": variant,
        "lock": encoder.lock,
        "clean_sha256": clean_hash,
        "n_clean": len(passages),
        "n_poison": 0,
        "dim": int(emb.shape[1]),
        "seconds": round(time.time() - t, 1),
    }, meta_path)
    log(f"  [{variant}] dense: done in {time.time() - t:.0f}s")


def build_dense_variant(cfg, encoder, variant):
    import faiss

    clean_dir = os.path.join(ic.index_dir(cfg, cfg["clean_variant"]), "dense")
    clean_meta = ic.read_json(os.path.join(clean_dir, "meta.json"))
    out = os.path.join(ic.index_dir(cfg, variant), "dense")
    meta_path = os.path.join(out, "meta.json")
    emb_path = os.path.join(out, "poison_embeddings.npy")
    poison_path = ic.poison_passages_path(cfg, variant)
    poison_hash = ic.file_sha256(poison_path)

    meta = check_lock(meta_path, encoder.lock, f"Dense index for {variant} ({out})")
    up_to_date = (
        meta
        and meta["poison_sha256"] == poison_hash
        and meta["clean_sha256"] == clean_meta["clean_sha256"]
        and os.path.exists(os.path.join(out, "index.faiss"))
    )
    if up_to_date:
        log(f"  [{variant}] dense: unchanged, reusing")
        return

    passages = ic.load_corpus(cfg, variant)
    poison = passages[clean_meta["n_clean"]:]

    if meta and meta["poison_sha256"] == poison_hash and os.path.exists(emb_path):
        poison_emb = np.load(emb_path)
        log(f"  [{variant}] dense: poisoned passages unchanged, reusing their embeddings")
    else:
        log(f"  [{variant}] dense: embedding {len(poison):,} poisoned passages")
        poison_emb = encoder.encode_passages(poison)
        os.makedirs(out, exist_ok=True)
        np.save(emb_path, poison_emb)

    clean_index = faiss.read_index(os.path.join(clean_dir, "index.faiss"))
    assert clean_index.ntotal == clean_meta["n_clean"]
    index = faiss.IndexFlatIP(clean_index.d)
    index.add(clean_index.reconstruct_n(0, clean_index.ntotal))
    index.add(poison_emb)
    faiss.write_index(index, os.path.join(out, "index.faiss"))
    write_ids(cfg, variant, [p["passage_id"] for p in passages])
    ic.write_json({
        "variant": variant,
        "lock": encoder.lock,
        "clean_sha256": clean_meta["clean_sha256"],
        "poison_sha256": poison_hash,
        "n_clean": clean_meta["n_clean"],
        "n_poison": len(poison),
        "dim": int(clean_index.d),
    }, meta_path)
    log(f"  [{variant}] dense: {index.ntotal:,} rows ({clean_meta['n_clean']:,} clean + {len(poison):,} poisoned)")


# ---------------------------------------------------------------- bm25

def build_bm25(cfg, variant):
    import bm25s

    out = os.path.join(ic.index_dir(cfg, variant), "bm25")
    meta_path = os.path.join(out, "meta.json")
    lock = ic.bm25_lock(cfg)
    hashes = {"clean_sha256": ic.file_sha256(ic.clean_passages_path(cfg))}
    if variant != cfg["clean_variant"]:
        hashes["poison_sha256"] = ic.file_sha256(ic.poison_passages_path(cfg, variant))

    meta = check_lock(meta_path, lock, f"BM25 index for {variant} ({out})")
    if meta and all(meta.get(k) == v for k, v in hashes.items()):
        log(f"  [{variant}] bm25: unchanged, reusing")
        return

    passages = ic.load_corpus(cfg, variant)
    log(f"  [{variant}] bm25: indexing {len(passages):,} passages")
    t = time.time()
    b = cfg["bm25"]
    tokens = ic.bm25_tokenize(cfg, [ic.format_passage(cfg, p) for p in passages], show_progress=False)
    retriever = bm25s.BM25(method=b["method"], k1=b["k1"], b=b["b"])
    retriever.index(tokens, show_progress=False)
    if os.path.exists(out):
        shutil.rmtree(out)
    retriever.save(out)
    write_ids(cfg, variant, [p["passage_id"] for p in passages])
    ic.write_json({"variant": variant, "lock": lock, "n_passages": len(passages), **hashes}, meta_path)
    log(f"  [{variant}] bm25: done in {time.time() - t:.0f}s")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=ic.DEFAULT_CONFIG)
    parser.add_argument("--variants", nargs="*", help="Poisoned corpora to index (default: all in the config).")
    parser.add_argument("--skip_dense", action="store_true")
    parser.add_argument("--skip_bm25", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Delete existing indexes for these corpora first.")
    args = parser.parse_args()

    cfg = ic.load_config(args.config)
    variants = args.variants if args.variants else cfg["variants"]
    corpora = [cfg["clean_variant"]] + [v for v in variants if v != cfg["clean_variant"]]

    for v in corpora[1:]:
        if not os.path.exists(ic.poison_passages_path(cfg, v)):
            raise SystemExit(f"Missing {ic.poison_passages_path(cfg, v)}")
    if not os.path.exists(ic.clean_passages_path(cfg)):
        raise SystemExit(f"Missing {ic.clean_passages_path(cfg)}; run src/data/build_clean_corpus.py first.")

    if args.rebuild:
        for v in corpora:
            if os.path.exists(ic.index_dir(cfg, v)):
                log(f"Removing {ic.index_dir(cfg, v)}")
                shutil.rmtree(ic.index_dir(cfg, v))

    log(f"Corpora: {', '.join(corpora)}")

    if not args.skip_dense:
        encoder = ic.make_encoder(cfg)
        log(f"Dense model: {encoder.lock['model_name']} @ {encoder.lock['revision']} on {getattr(encoder, 'device', '?')}")
        build_dense_clean(cfg, encoder)
        for v in corpora[1:]:
            build_dense_variant(cfg, encoder, v)

    if not args.skip_bm25:
        for v in corpora:
            build_bm25(cfg, v)

    log("All indexes ready.")


if __name__ == "__main__":
    main()
