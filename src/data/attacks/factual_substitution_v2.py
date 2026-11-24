"""Factual substitution, version 2: poisons the number a question asks about.

Same attack as factual_substitution.py (v1): the first number in the gold answer is
replaced, in the first gold passage that contains it, by a different value, and the
poisoned passage is added 3 times. Version 2 fixes the v1 defects:

- the answer's number is read without a trailing comma ("30, 2017" gives 30, not "30,");
- every standalone occurrence is replaced, including one followed by a comma or a full
  stop; digits that belong to a larger number ("1,000") or a decimal ("1.5") are left alone;
- a number that is part of a decimal in the answer is not attacked;
- a day of the month stays a valid day (1-28); a year (1000-2030) is shifted by 5-20 years
  and never past 2026; a decade ("1970s") moves by whole decades; an ordinal suffix is
  rewritten to match the new number ("21st" -> "34th");
- a number that is part of a range ("26-30 August", "2013-14", "aged 16-21") or a
  hyphenated code ("1-800-...") in the passage is not attacked, because changing one end
  breaks the range;
- the false value must not already appear in the original passage;
- every poisoned passage is checked before it is written: the false value is present,
  the true value is absent and no date is impossible. A question that fails is rejected
  with its reason.

For questions that v1 handled correctly, v2 draws the same false value as v1 (same
seed, same random sequence), so the two versions differ only where v1 was wrong.
"""

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter

LAST_YEAR = 2026
MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
          "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
# a number in text: thousands with commas ("1,500,000") or a plain run of digits
NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)|\d+")
ORDINAL = re.compile(r"(st|nd|rd|th)\b", re.IGNORECASE)


def v1_would_draw(row: dict) -> bool:
    """True if factual_substitution.py (v1) consumes one random draw for this question."""
    m = re.search(r"\d[\d,]*", row["answer"])
    if not m:
        return False
    pat = re.compile(rf"(?<![\d,]){re.escape(m.group())}(?![\d,])")
    gold = set(row["supporting_facts"]["title"])
    return any(t in gold and pat.search("".join(s))
               for t, s in zip(row["context"]["title"], row["context"]["sentences"]))


def answer_number(answer: str):
    """First number in the answer and whether it is part of a decimal."""
    m = NUMBER.search(answer)
    if not m:
        return None, False
    before = answer[m.start() - 1] if m.start() > 0 else ""
    after = answer[m.end():m.end() + 2]
    decimal = (before == "." and m.start() > 1 and answer[m.start() - 2].isdigit()) or \
              (len(after) == 2 and after[0] == "." and after[1].isdigit())
    return m.group(), decimal


def occurrence_pattern(token: str) -> re.Pattern:
    """A standalone occurrence of the number: not part of a larger number or a decimal."""
    return re.compile(rf"(?<![\d.,]){re.escape(token)}(?!\d|[.,]\d)")


def day_contexts(text: str, token: str) -> bool:
    """True if some occurrence of the number is a day of the month."""
    t = re.escape(token)
    return bool(re.search(rf"\b(?:{MONTHS})\.? {t}(?!\d)", text) or
                re.search(rf"(?<![\d.,]){t}(?:st|nd|rd|th)? (?:of )?(?:{MONTHS})\b", text))


def is_year(token: str) -> bool:
    return len(token) == 4 and token.isdigit() and 1000 <= int(token) <= 2030


def v1_is_year(token: str) -> bool:
    return len(token.replace(",", "")) == 4 and 1500 <= int(token.replace(",", "")) <= 2030


def in_range(text: str, token: str) -> bool:
    """True if some occurrence is one end of a range or part of a hyphenated code."""
    t = re.escape(token)
    return bool(re.search(rf"(?<![\d.,]){t}\s?[–—-]\s?\d", text) or
                re.search(rf"\d\s?[–—-]\s?{t}(?!\d|[.,]\d)", text))


def decade(text: str, token: str) -> bool:
    return bool(re.search(rf"(?<![\d.,]){re.escape(token)}s\b", text))


def v1_draw(value: int, token: str, rng: random.Random) -> int:
    """The exact draw v1 makes (one rng.choice call)."""
    if v1_is_year(token):
        shifted = value + rng.choice([-20, -15, -10, -5, 5, 10, 15, 20])
    else:
        shifted = max(1, round(value * rng.choice([0.55, 0.7, 1.35, 1.6, 2.0])))
    return value + 1 if shifted == value else shifted


def fmt(value: int, token: str) -> str:
    return f"{value:,}" if "," in token else str(value)


def ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def choose_value(token: str, text: str, first_draw, qrng: random.Random):
    """Pick the false value: v1's draw if it is valid here, otherwise a valid redraw."""
    value = int(token.replace(",", ""))
    day = value <= 31 and day_contexts(text, token)
    dec = is_year(token) and decade(text, token)
    if is_year(token) and not v1_is_year(token):
        first_draw = None          # v1 treated this year as a plain number
    present = {m.group().replace(",", "") for m in NUMBER.finditer(text)}

    def ok(v):
        if v == value or v < 1 or str(v) in present:
            return False
        if day and v > 28:
            return False
        if is_year(token) and (v > LAST_YEAR or not 5 <= abs(v - value) <= 20):
            return False
        if dec and (v - value) % 10:
            return False
        return True

    candidates = [first_draw] if first_draw is not None else []
    if is_year(token):
        candidates += [value + d for d in qrng.sample([-20, -15, -10, -5, 5, 10, 15, 20], 8)]
    elif day:
        candidates += qrng.sample(range(1, 29), 28)
    else:
        candidates += [max(1, round(value * f)) for f in qrng.sample([0.55, 0.7, 1.35, 1.6, 2.0], 5)]
        candidates += [value + d for d in (1, 2, 3, -1, -2, 4, 5)]
    for v in candidates:
        if v is not None and ok(v):
            return v, day, (v != first_draw)
    return None, day, True


def replace_all(text: str, token: str, new: str) -> tuple[str, int]:
    """Replace every standalone occurrence; rewrite a following ordinal suffix."""
    pat = occurrence_pattern(token)
    n_new = int(new.replace(",", ""))

    out, count, pos = [], 0, 0
    for m in pat.finditer(text):
        out.append(text[pos:m.start()])
        out.append(new)
        count += 1
        pos = m.end()
        suf = ORDINAL.match(text, pos)
        if suf:
            s = ordinal_suffix(n_new)
            out.append(s.upper() if suf.group().isupper() else s)
            pos = suf.end()
    out.append(text[pos:])
    return "".join(out), count


def impossible_dates(text: str) -> list[str]:
    bad = []
    for m in re.finditer(rf"\b(?:{MONTHS})\.? (\d{{1,2}})(?!\d)(?!,\d)", text):
        if not 1 <= int(m.group(1)) <= 31:
            bad.append(m.group())
    for m in re.finditer(rf"(?<![\d.,])(\d{{1,2}})(?:st|nd|rd|th)? (?:of )?(?:{MONTHS})\b", text):
        if not 1 <= int(m.group(1)) <= 31:
            bad.append(m.group())
    return bad


def standalone(text: str, token: str) -> bool:
    return bool(occurrence_pattern(token).search(text))


def passage_id(title: str, salt: str) -> str:
    return "p_" + hashlib.sha256(f"{title}::{salt}".encode("utf-8")).hexdigest()[:12]


def build_attack(row: dict, rng: random.Random, seed: int, copies: int):
    """Return (passages, provenance, None) or (None, None, reason)."""
    draw_from_v1 = v1_would_draw(row)
    token, decimal = answer_number(row["answer"])
    # v1 consumed exactly one draw for these questions; consume it here too, in the same order
    first = None
    if draw_from_v1:
        v1_token = re.search(r"\d[\d,]*", row["answer"]).group()
        first = v1_draw(int(v1_token.replace(",", "")), v1_token, rng)
    if token is None:
        return None, None, "answer_has_no_number"
    if decimal:
        return None, None, "answer_number_is_part_of_a_decimal"

    pat = occurrence_pattern(token)
    gold = set(row["supporting_facts"]["title"])
    target = None
    for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
        if title in gold and pat.search("".join(sents)):
            target = (title, "".join(sents))
            break
    if target is None:
        return None, None, "number_absent_from_gold_passage"
    title, text = target
    if in_range(text, token):
        return None, None, "number_is_part_of_a_range"

    qrng = random.Random(f"{seed}::{row['id']}")
    value, day, redrawn = choose_value(token, text, first, qrng)
    if value is None:
        return None, None, "no_valid_false_value"
    new = fmt(value, token)
    poisoned, n_occ = replace_all(text, token, new)

    # final check, same rule as the LLM attacks: false value present, true value absent
    if not standalone(poisoned, new):
        return None, None, "check:false_value_absent"
    if standalone(poisoned, token):
        return None, None, "check:true_value_still_present"
    if impossible_dates(poisoned) and not impossible_dates(text):
        return None, None, "check:impossible_date"
    if poisoned == text:
        return None, None, "check:passage_unchanged"

    others = sum(1 for t, s in zip(row["context"]["title"], row["context"]["sentences"])
                 if t != title and standalone("".join(s), token))
    passages, provenance = [], []
    for c in range(copies):
        pid = passage_id(title, salt=f"factual_v2::{row['id']}::{c}")
        passages.append({"passage_id": pid, "title": title, "text": poisoned})
        provenance.append({
            "question_id": row["id"], "question_type": row["type"],
            "attack_type": "factual_substitution", "attack_version": "v2",
            "source_passage_title": title, "original_value": token, "poisoned_value": new,
            "poisoned_passage_id": pid, "copy_index": c,
            "occurrences_replaced": n_occ, "day_of_month": day, "differs_from_v1_draw": bool(redrawn),
            "residual_nongold_passages_with_answer": others,
        })
    return passages, provenance, None


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description="Factual substitution attack, version 2.")
    p.add_argument("--raw_path", default="data/raw/dev.jsonl")
    p.add_argument("--out_dir", default="data/processed/poisoned_corpus/factual_v2")
    p.add_argument("--copies_per_question", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    passages, provenance, rejected, outcomes = [], [], [], Counter()
    with open(args.raw_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ps, pv, reason = build_attack(row, rng, args.seed, args.copies_per_question)
            if ps is None:
                outcomes[reason] += 1
                if reason not in ("answer_has_no_number", "number_absent_from_gold_passage"):
                    rejected.append({"question_id": row["id"], "answer": row["answer"], "reason": reason})
                continue
            passages += ps
            provenance += pv
            outcomes["poisoned"] += 1
            outcomes[f"poisoned_{row['type']}"] += 1

    assert len(passages) == outcomes["poisoned"] * args.copies_per_question
    assert len({x["passage_id"] for x in passages}) == len(passages), "duplicate passage IDs"
    write_jsonl(passages, os.path.join(args.out_dir, "poisoned_passages.jsonl"))
    write_jsonl(provenance, os.path.join(args.out_dir, "provenance.jsonl"))
    write_jsonl(rejected, os.path.join(args.out_dir, "rejected.jsonl"))
    manifest = {
        "attack": "factual_substitution", "version": "v2", "script": "src/data/attacks/factual_substitution_v2.py",
        "raw_path": args.raw_path, "seed": args.seed, "copies_per_question": args.copies_per_question,
        "last_year": LAST_YEAR, "outcomes": dict(sorted(outcomes.items())),
        "poisoned_questions": outcomes["poisoned"], "poisoned_passages": len(passages),
        "false_value_redrawn": sum(1 for r in provenance if r["copy_index"] == 0 and r["differs_from_v1_draw"]),
    }
    with open(os.path.join(args.out_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
