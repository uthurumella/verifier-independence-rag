"""Poisons the number a question asks about, leaving the rest of the passage alone.

Only replaces a number if it also appears in the gold answer, so the attack
hits the fact the answer depends on.
"""

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter

NUMBER_IN_TEXT = re.compile(r"\d[\d,]*")


def extract_answer_number(answer: str) -> str | None:
    """Return the first number token in the answer, as written. None if absent."""
    match = NUMBER_IN_TEXT.search(answer)
    return match.group() if match else None


def perturb(number_token: str, rng: random.Random) -> str:
    """Return a plausible but different value, preserving the original's comma style."""
    value = int(number_token.replace(",", ""))

    if 1500 <= value <= 2030 and len(number_token.replace(",", "")) == 4:
        shifted = value + rng.choice([-20, -15, -10, -5, 5, 10, 15, 20])
    else:
        factor = rng.choice([0.55, 0.7, 1.35, 1.6, 2.0])
        shifted = max(1, round(value * factor))

    if shifted == value:
        shifted = value + 1

    return f"{shifted:,}" if "," in number_token else str(shifted)


def locate_target(row: dict, number_token: str):
    """Find the first gold passage containing the number. Returns (title, sentences) or None."""
    pattern = re.compile(rf"(?<![\d,]){re.escape(number_token)}(?![\d,])")
    gold_titles = set(row["supporting_facts"]["title"])
    titles = row["context"]["title"]
    sentence_lists = row["context"]["sentences"]

    for title, sentences in zip(titles, sentence_lists):
        if title in gold_titles and pattern.search("".join(sentences)):
            return title, sentences, pattern
    return None


def passage_id(title: str, salt: str) -> str:
    digest = hashlib.sha256(f"{title}::{salt}".encode("utf-8")).hexdigest()
    return f"p_{digest[:12]}"


def build_attack(row: dict, rng: random.Random, copies: int):
    """Return (passages, provenance) for one question, or (None, skip_reason)."""
    number_token = extract_answer_number(row["answer"])
    if number_token is None:
        return None, "answer_has_no_number"

    target = locate_target(row, number_token)
    if target is None:
        return None, "number_absent_from_gold_passage"

    title, sentences, pattern = target
    replacement = perturb(number_token, rng)
    poisoned_text = pattern.sub(replacement, "".join(sentences))

    passages, provenance = [], []
    for copy_index in range(copies):
        pid = passage_id(title, salt=f"factual::{row['id']}::{copy_index}")
        passages.append({"passage_id": pid, "title": title, "text": poisoned_text})
        provenance.append({
            "question_id": row["id"],
            "question_type": row["type"],
            "attack_type": "factual_substitution",
            "source_passage_title": title,
            "original_value": number_token,
            "poisoned_value": replacement,
            "poisoned_passage_id": pid,
            "copy_index": copy_index,
        })
    return passages, provenance


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build an answer-anchored factual-substitution corpus.")
    parser.add_argument("--raw_path", default="data/raw/dev.jsonl")
    parser.add_argument("--out_dir", default="data/processed/poisoned_corpus/factual_v1")
    parser.add_argument("--copies_per_question", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    all_passages, all_provenance = [], []
    outcomes = Counter()

    with open(args.raw_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            passages, result = build_attack(row, rng, args.copies_per_question)
            if passages is None:
                outcomes[result] += 1
                continue
            all_passages.extend(passages)
            all_provenance.extend(result)
            outcomes["poisoned"] += 1
            outcomes[f"poisoned_{row['type']}"] += 1

    write_jsonl(all_passages, os.path.join(args.out_dir, "poisoned_passages.jsonl"))
    write_jsonl(all_provenance, os.path.join(args.out_dir, "provenance.jsonl"))

    total = sum(v for k, v in outcomes.items() if not k.startswith("poisoned_"))
    print(f"Questions read:            {total}")
    print(f"  poisoned:                {outcomes['poisoned']}")
    print(f"    bridge:                {outcomes['poisoned_bridge']}")
    print(f"    comparison:            {outcomes['poisoned_comparison']}")
    print(f"  skipped (no number):     {outcomes['answer_has_no_number']}")
    print(f"  skipped (not in gold):   {outcomes['number_absent_from_gold_passage']}")
    print(f"\nPoisoned passages written: {len(all_passages)}")

    assert len(all_passages) == outcomes["poisoned"] * args.copies_per_question
    assert len({p["passage_id"] for p in all_passages}) == len(all_passages), "duplicate passage IDs"
    print("Sanity checks passed: counts consistent, passage IDs unique.")

    print("\n--- Sample ---")
    for record in all_provenance[:6:3]:
        print(f"[{record['question_type']}] {record['original_value']} -> {record['poisoned_value']}  ({record['source_passage_title']})")


if __name__ == "__main__":
    main()
