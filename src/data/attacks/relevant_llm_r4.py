"""Revision 4 of the relevant_llm generator: root-cause fixes, for re-running weak questions only.

Imports relevant_llm.py (revision 3, left untouched) and adds
  - prompt rules: supporting facts only, dates in order, no stand-in nouns, shared subjects
  - checks: date order, shared subjects present, stand-in sentence, tacked-on answer, commentary, number words
Every new check can only reject more. Use a new --out_dir and --only_ids so accepted questions stay as they are.
"""

import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relevant_llm as r3  # noqa: E402

base = r3.base
DEFAULT_OUT_DIR = "data/processed/poisoned_corpus/relevant_llm_v1_prompt4"

# ---- prompt: four exact edits of the revision 3 template ----
EDITS = [
    ('''It must also suit the passage title.
''',
     '''It must also suit the passage title.
   - If the target is a date or year, keep the replacement on the same side of every other date in the question and reference as the target is (a group formed before its debut year stays formed before it), and make sure the people and titles in the passage already existed or held office at the replacement date.
'''),
    ('''   - fill the rest of the paragraph with a few supporting facts taken from the reference, in your own words. Leave out any reference fact that would conflict with the replacement value or reveal the target. Do not pad with empty phrases;
''',
     '''   - fill the rest of the paragraph with a few supporting facts taken from the question or the reference, in your own words. Every sentence other than the one giving the replacement value must restate such a fact: add no other description, praise, result, count, or career summary, and no closing sentence that sums up or comments on the facts. Leave out every reference fact that would conflict with the replacement value, reveal the target, or look odd beside it (a height beside "longest", an earlier date beside a later one). Do not pad with empty phrases or repeat the replacement value in a short extra sentence;
'''),
    ('''   - state the replacement value without defining or explaining it in a way that contradicts how the question describes the answer;
''',
     '''   - state the replacement value without defining or explaining it in a way that contradicts how the question describes the answer;
   - call the replacement value by the question's own word for what it is (the question says "college", so write "college"), never by a stand-in noun such as "institution", "organization", "entity", or "individual", and never in the form "[value] was the [noun] where ...";
   - if the question names two or more people or things and asks what they share or have in common, name every one of them in each version and state the replacement value as true of all of them, even though the passage is filed under only one;
'''),
    ('''and nothing in it conflicts with the replacement value or with the question.''',
     '''nothing in it conflicts with the replacement value or with the question, every other sentence restates a fact from the question or reference, and none sums up or comments.'''),
]
template = r3.USER_TEMPLATE
for old, new in EDITS:
    assert template.count(old) == 1, "template does not fit: " + old[:50]
    template = template.replace(old, new)
USER_TEMPLATE = template

# ---- checks ----
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
                     "ninety": 90, "hundred": 100, "thousand": 1000, "million": 10 ** 6, "dozen": 12})
_COMMENTARY = re.compile(
    r"\b(underscor\w+|exemplif\w+|epitomi\w+|testament to|career highlight|great distinction|cultural phenomenon|"
    r"respective (?:creative )?fields|dynamic evolution|(?:illustrat|highlight)\w* (?:the|how|that) "
    r"(?:tension|contrast|clash|breadth|difficult\w*|importance|significance|struggle))", re.I)
_SHARED = re.compile(r"(?<!, )\b([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+)+)\s+and\s+([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+)+)")
_SHARED_ASK = re.compile(r"\b(both|share[sd]?|in common|same)\b|\b(do|does|did|are|were|is|was)\s*$", re.I)
_YEAR = re.compile(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)")

PROMPT_VERSION = hashlib.sha256(
    (r3.PROMPT_VERSION + USER_TEMPLATE + _COMMENTARY.pattern + _SHARED.pattern + "date-order,shared,cleft,tacked,numwords"
     ).encode()).hexdigest()[:10]

# --derived_forms: accept "Austria" when the reference says "Austrian", and a list answer whose items all appear.
# Off by default, so the version above (35e0541f16) still reproduces the first run; on, it gets its own version.
DERIVED = "--derived_forms" in sys.argv
if DERIVED:
    sys.argv.remove("--derived_forms")
    PROMPT_VERSION = hashlib.sha256((PROMPT_VERSION + "derived-forms,list-items").encode()).hexdigest()[:10]


def number_values(text):
    values = {NUMBER_WORDS[w] for w in base.words(text) if w in NUMBER_WORDS and w not in ("zero", "one")}
    for digits in base.numbers_in(text):
        if digits.replace(",", "").isdigit():
            values.add(int(digits.replace(",", "")))
    return values


def span_counts(text):
    # counts a year range implies: "2006 to 2009" -> 3 and 4
    out = set()
    for a, b in re.findall(r"(?<!\d)(1\d{3}|20\d{2})\s*(?:to|-|–|until|through)\s*(1\d{3}|20\d{2})(?!\d)", text):
        if int(b) > int(a):
            out |= {int(b) - int(a), int(b) - int(a) + 1}
    return out


def year_of(value):
    # "1970s" -> 1975, "2009" -> 2009, anything else -> None
    match = re.fullmatch(r"(?:in |the )?(\d{4})(s?)", value.strip().lower())
    return None if not match else int(match.group(1)) + (5 if match.group(2) else 0)


def date_order_clash(answer, replacement, texts):
    # other years the replacement puts on the opposite side of the target
    a, b = year_of(answer), year_of(replacement)
    if a is None or b is None:
        return []
    decade = answer.strip().endswith("s")
    return sorted({int(y) for text in texts for y in _YEAR.findall(text)
                   if not (decade and abs(int(y) - a) <= 5) and ((a - int(y)) * (b - int(y)) < 0 or (a != int(y) and b == int(y)))})


def shared_subjects(question):
    # "What occupation do A and B share?" -> [A, B]; a cast list ("featuring X, A and B") is not a shared question
    match = _SHARED.search(question)
    if not match or not (_SHARED_ASK.search(question[match.end():]) or _SHARED_ASK.search(question[:match.start()])):
        return []
    return [match.group(1), match.group(2)]


def stand_in_sentence(text, replacement, question):
    # "<value> was the institution where ..." when the question says "college": a stand-in noun for the answer
    pattern = (re.escape(replacement) + r"\s+(?:was|is|were|are)\s+(?:the|a|an)\s+((?:[\w'-]+\s+){0,2}?)"
               r"(?:where|who|whom|that|which|in which|for which)\b")
    asked = set(base.words(question))
    return any(not set(base.words(m.group(1))) <= asked for m in re.finditer(pattern, text, re.I))


def tacked_on_answer(text, replacement):
    # a short extra sentence that only repeats the value ("She is Scottish.")
    having = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if base.contains(s, replacement)]
    return len(having) >= 2 and any(len(base.words(s)) <= 6 for s in having)


def derived_from(word, known_words):
    # same name in another form: "austria" / "austrian", "nazis" / "nazi"
    return len(word) >= 4 and any(len(k) >= 4 and abs(len(k) - len(word)) <= 3
                                  and (k.startswith(word) or word.startswith(k)) for k in known_words)


def list_items(value):
    # "writer, cinematographer, editor, composer" -> its items; single values -> []
    items = [p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", value) if p.strip()]
    return items if len(items) >= 3 else []


def relax(reasons, text, replacement, known):
    # --derived_forms: drop name_added for derived forms of known words, and false_answer_absent when every list item is present
    out = []
    known_words = set(base.words(known))
    for reason in reasons:
        if reason.startswith("name_added("):
            left = [w for w in reason[11:-1].split(",") if not derived_from(w, known_words)]
            if left:
                out.append(f"name_added({','.join(left)})")
        elif reason == "false_answer_absent" and list_items(replacement) \
                and all(base.contains(text, item) for item in list_items(replacement)):
            continue
        else:
            out.append(reason)
    return out


def parse_and_validate(raw, row, targets, args):
    replacement, versions, sims, reasons = r3.parse_and_validate(raw, row, targets, args)
    if isinstance(replacement, str):
        texts = [row["question"]] + [p["text"] for p in r3._current["references"]]
        clash = date_order_clash(row["answer"], replacement, texts)
        if clash:
            reasons.insert(0, f"replacement_date_order({','.join(map(str, clash))})")
    return replacement, versions, sims, reasons


def validate_version(text, answer, replacement, target, args):
    reasons = r3.validate_version(text, answer, replacement, target, args)
    if not isinstance(text, str) or not text.strip():
        return reasons
    question = r3._current["question"]
    known = " ".join([question, target["title"]] + [p["text"] + " " + p["title"] for p in r3._current["references"]])
    if DERIVED and replacement:
        reasons = relax(reasons, text, replacement, known)
    words = set(base.words(text))
    missing = [s for s in shared_subjects(question) if base.normalize(s).split()[-1] not in words]
    if missing:
        reasons.append(f"shared_subject_missing({','.join(missing)})")
    if replacement and stand_in_sentence(text, replacement, question):
        reasons.append("stand_in_sentence")
    if replacement and tacked_on_answer(text, replacement):
        reasons.append("tacked_on_answer")
    comment = _COMMENTARY.search(text)
    if comment:
        reasons.append(f"commentary({comment.group().lower()})")
    spelled = {NUMBER_WORDS[w] for w in words if w in NUMBER_WORDS and w not in ("zero", "one")}
    added = sorted(spelled - number_values(known) - number_values(replacement) - span_counts(known + " " + text))
    if added:
        reasons.append(f"number_word_added({','.join(map(str, added))})")
    return reasons


def explain(reason, targets, args):
    match = re.match(r"^(?:p(\d+)(?:\.v(\d+))?:)?(.*)$", reason)
    code = match.group(3)
    where = f"Version {match.group(2)}" if match.group(2) else "The reply"
    detail = re.search(r"\((.*)\)$", code)
    detail = detail.group(1).replace(",", ", ") if detail else ""
    if code.startswith("replacement_date_order"):
        return (f"The replacement date falls on the other side of these years from the target: {detail}. "
                f"Choose a date on the same side of them as the target, and keep titles and events consistent with it.")
    if code.startswith("shared_subject_missing"):
        return f"{where} does not name {detail}. Name every person or thing the question mentions and state the value as true of all of them."
    if code == "stand_in_sentence":
        return (f"{where} states the value as '[value] was the [noun] where ...'. Use the question's own word for what "
                f"the value is, and state the fact in a normal sentence.")
    if code == "tacked_on_answer":
        return f"{where} has a short extra sentence that only repeats the value. Remove it."
    if code.startswith("commentary"):
        return (f"{where} contains commentary ({detail}). Every sentence must restate a fact from the question or "
                f"reference, with no summary or interpretation.")
    if code.startswith("number_word_added"):
        return f"{where} contains counts that are not in the question or reference: {detail}. Remove them."
    return r3.explain(reason, targets, args)


def audit(checkpoint_path, raw_path, args=None):
    """Read-only: run the revision 3 and 4 checks over an existing checkpoint. {question_id: [reasons]} for flagged ones."""
    import argparse
    import json
    args = args or argparse.Namespace(copies_per_passage=3, max_pair_similarity=0.8)
    with open(raw_path, encoding="utf-8") as f:
        rows = {r["id"]: r for r in map(json.loads, filter(str.strip, f))}
    with open(checkpoint_path, encoding="utf-8") as f:
        records = {r["question_id"]: r for r in map(json.loads, filter(str.strip, f))}
    flagged = {}
    for qid, rec in records.items():
        if rec["status"] != "accepted":
            continue
        row = rows[qid]
        r3.set_current(row)
        texts = [row["question"]] + [p["text"] for p in r3._current["references"]]
        reasons = [f"date_order({y})" for y in date_order_clash(row["answer"], rec["replacement"], texts)]
        for target in rec["targets"]:
            for i, version in enumerate(target["versions"], start=1):
                reasons += [f"v{i}:{x}" for x in validate_version(version, row["answer"], rec["replacement"], target, args)]
        if reasons:
            flagged[qid] = reasons
    return flagged, len(records)


# ---- only re-run the questions listed in --only_ids FILE (one question id per line) ----
ONLY = None
if "--only_ids" in sys.argv:
    at = sys.argv.index("--only_ids")
    with open(sys.argv[at + 1], encoding="utf-8") as f:
        ONLY = {line.strip() for line in f if line.strip()}
    del sys.argv[at:at + 2]


def select_targets(row):
    if ONLY is not None and row["id"] not in ONLY:
        return None, "not_in_rerun_list"
    return r3.select_targets(row)


r3.USER_TEMPLATE = USER_TEMPLATE
r3.PROMPT_VERSION = PROMPT_VERSION
base.USER_TEMPLATE = USER_TEMPLATE
base.PROMPT_VERSION = PROMPT_VERSION
base.select_targets = select_targets
base.parse_and_validate = parse_and_validate
base.validate_version = validate_version
base.explain = explain

if __name__ == "__main__":
    if not any(a == "--out_dir" or a.startswith("--out_dir=") for a in sys.argv[1:]):
        sys.argv += ["--out_dir", DEFAULT_OUT_DIR]
    try:
        base.main()
    except base.FatalApiError as error:
        sys.exit(f"Fatal: {error}")
