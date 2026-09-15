"""Splits the HotpotQA dev pool into our own dev and test sets."""

import argparse
import json
import os
import random

import yaml


def load_question_ids(raw_path):
    ids = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["id"] if "id" in row else row["_id"])
    return ids


def write_id_list(ids, path):
    with open(path, "w", encoding="utf-8") as f:
        for qid in ids:
            f.write(json.dumps({"question_id": qid}) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build a fixed dev/test split at the question level.")
    parser.add_argument("--raw_path", type=str, default="data/raw/dev.jsonl")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--out_dir", type=str, default="data/processed/splits")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["project_split"]["seed"]
    dev_ratio = cfg["project_split"]["dev_ratio"]

    question_ids = load_question_ids(args.raw_path)
    total = len(question_ids)
    print(f"Loaded {total} question IDs from {args.raw_path}")

    rng = random.Random(seed)
    shuffled = question_ids.copy()
    rng.shuffle(shuffled)

    split_point = int(total * dev_ratio)
    dev_ids = shuffled[:split_point]
    test_ids = shuffled[split_point:]

    os.makedirs(args.out_dir, exist_ok=True)
    dev_path = os.path.join(args.out_dir, cfg["output"]["dev_ids_file"])
    test_path = os.path.join(args.out_dir, cfg["output"]["test_ids_file"])

    write_id_list(dev_ids, dev_path)
    write_id_list(test_ids, test_path)

    print(f"Seed: {seed} | dev_ratio: {dev_ratio}")
    print(f"  -> wrote {len(dev_ids)} question IDs to {dev_path}")
    print(f"  -> wrote {len(test_ids)} question IDs to {test_path}")

    overlap = set(dev_ids) & set(test_ids)
    assert len(overlap) == 0, f"ERROR: {len(overlap)} question IDs appear in both splits!"
    print("Sanity check passed: no overlap between dev and test splits.")


if __name__ == "__main__":
    main()
