"""Builds a new corpus by adding one attack's poisoned passages to the clean pool.

The clean corpus is read but never modified, so it stays available as the
baseline. Each attack type gets its own corpus this way, with the same clean
passages underneath.
"""

import argparse
import json
import os


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Merge clean and poisoned passages into one corpus.")
    parser.add_argument("--clean_path", default="data/processed/clean_corpus/passages.jsonl")
    parser.add_argument("--poison_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--variant_name", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    clean = read_jsonl(args.clean_path)
    poison = read_jsonl(args.poison_path)

    clean_ids = {p["passage_id"] for p in clean}
    poison_ids = {p["passage_id"] for p in poison}

    collisions = clean_ids & poison_ids
    assert not collisions, f"passage_id collision between clean and poison: {len(collisions)} IDs"
    assert len(poison_ids) == len(poison), "duplicate passage_id within poisoned passages"

    corpus = clean + poison
    corpus_path = os.path.join(args.out_dir, "corpus.jsonl")
    write_jsonl(corpus, corpus_path)

    shared_titles = {p["title"] for p in poison} & {p["title"] for p in clean}

    manifest = {
        "variant_name": args.variant_name,
        "clean_source": args.clean_path,
        "poison_source": args.poison_path,
        "clean_passages": len(clean),
        "poisoned_passages": len(poison),
        "total_passages": len(corpus),
        "titles_shared_with_clean": len(shared_titles),
    }
    manifest_path = os.path.join(args.out_dir, "corpus_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    assert len(corpus) == len(clean) + len(poison)
    assert len({p["passage_id"] for p in corpus}) == len(corpus), "duplicate passage_id in merged corpus"

    print(f"Variant:              {args.variant_name}")
    print(f"Clean passages:       {len(clean):,}")
    print(f"Poisoned passages:    {len(poison):,}")
    print(f"Total corpus:         {len(corpus):,}")
    print(f"Titles shared:        {len(shared_titles):,}  (expected -- poison reuses gold titles)")
    print(f"\nWrote corpus to:      {corpus_path}")
    print(f"Wrote manifest to:    {manifest_path}")
    print("Sanity checks passed: no ID collisions, no duplicates, counts consistent.")


if __name__ == "__main__":
    main()
