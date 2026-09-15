"""Downloads HotpotQA (distractor setting) and saves train and dev as jsonl."""

import argparse
import json
import os
from datetime import datetime, timezone

from datasets import load_dataset


def write_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_passages(dataset):
    """Count total context paragraphs and gold supporting passages across a split."""
    total_context_paragraphs = 0
    total_gold_titles = 0
    for row in dataset:
        titles = row["context"]["title"]
        total_context_paragraphs += len(titles)
        total_gold_titles += len(set(row["supporting_facts"]["title"]))
    return total_context_paragraphs, total_gold_titles


def main():
    parser = argparse.ArgumentParser(description="Download and structure the full HotpotQA dataset.")
    parser.add_argument("--out_dir", type=str, default="data/raw", help="Output directory for raw jsonl files.")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "dev"],
        help="Which split(s) to download.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="distractor",
        choices=["distractor", "fullwiki"],
        help="HotpotQA config. Use 'distractor' for the gold+distractor setting used in this project.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    splits_to_load = ["train", "validation"] if args.split == "all" else (
        ["train"] if args.split == "train" else ["validation"]
    )

    stats = {
        "source": "hotpotqa/hotpot_qa",
        "config": args.config,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY-SA 4.0 (data), Apache 2.0 (original code)",
        "splits": {},
    }

    for split_name in splits_to_load:
        print(f"Loading HotpotQA [{args.config}] split={split_name} ...")
        ds = load_dataset("hotpotqa/hotpot_qa", args.config, split=split_name)

        out_name = "dev" if split_name == "validation" else split_name
        out_path = os.path.join(args.out_dir, f"{out_name}.jsonl")
        write_jsonl(ds, out_path)

        n_questions = len(ds)
        n_context_paragraphs, n_gold_titles = count_passages(ds)

        stats["splits"][out_name] = {
            "num_questions": n_questions,
            "total_context_paragraphs": n_context_paragraphs,
            "avg_paragraphs_per_question": round(n_context_paragraphs / n_questions, 2) if n_questions else 0,
            "total_gold_supporting_titles": n_gold_titles,
        }

        print(f"  -> wrote {n_questions} questions to {out_path}")

    card_path = os.path.join(args.out_dir, "dataset_card.json")
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote dataset card to {card_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
