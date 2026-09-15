"""Flattens HotpotQA's per-question passages into one deduplicated pool to retrieve over."""

import argparse
import hashlib
import json
import os


def passage_id(title: str) -> str:
    """Stable ID derived from the passage title. Same title -> same ID, always."""
    digest = hashlib.sha256(title.encode("utf-8")).hexdigest()
    return f"p_{digest[:12]}"


def main():
    parser = argparse.ArgumentParser(description="Build the flat, deduplicated clean passage corpus.")
    parser.add_argument("--raw_path", type=str, default="data/raw/dev.jsonl")
    parser.add_argument("--out_dir", type=str, default="data/processed/clean_corpus")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    passages = {}          # passage_id -> {title, text}
    question_map_rows = []  # one row per (question, passage) pair

    n_questions = 0
    with open(args.raw_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_questions += 1

            question_id = row["id"]
            gold_titles = set(row["supporting_facts"]["title"])

            titles = row["context"]["title"]
            sentence_lists = row["context"]["sentences"]

            for position, (title, sents) in enumerate(zip(titles, sentence_lists)):
                pid = passage_id(title)

                if pid not in passages:
                    passages[pid] = {
                        "passage_id": pid,
                        "title": title,
                        "text": "".join(sents),
                    }

                question_map_rows.append({
                    "question_id": question_id,
                    "passage_id": pid,
                    "position": position,
                    "is_gold": title in gold_titles,
                })

    passages_path = os.path.join(args.out_dir, "passages.jsonl")
    with open(passages_path, "w", encoding="utf-8") as f:
        for p in passages.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    map_path = os.path.join(args.out_dir, "question_passage_map.jsonl")
    with open(map_path, "w", encoding="utf-8") as f:
        for r in question_map_rows:
            f.write(json.dumps(r) + "\n")

    n_gold_rows = sum(1 for r in question_map_rows if r["is_gold"])

    print(f"Questions processed:        {n_questions}")
    print(f"Unique passages (deduped):  {len(passages)}")
    print(f"Question-passage links:     {len(question_map_rows)}")
    print(f"  of which gold:            {n_gold_rows}")
    print(f"\nWrote passages to:  {passages_path}")
    print(f"Wrote map to:       {map_path}")

    # Sanity check: every passage_id in the map must exist in passages
    passage_ids = set(passages.keys())
    missing = [r["passage_id"] for r in question_map_rows if r["passage_id"] not in passage_ids]
    assert not missing, f"ERROR: {len(missing)} map rows reference unknown passage IDs"
    print("Sanity check passed: every mapped passage_id exists in passages.jsonl")


if __name__ == "__main__":
    main()
