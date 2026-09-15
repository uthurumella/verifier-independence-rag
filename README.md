# Can We Trust the Verifier? Verification Independence in RAG Under Knowledge-Base Poisoning

## Overview

A common way to make retrieval-augmented generation safer is to add a verifier:
a second model that checks the generated answer before it is returned. This
project asks whether that check does anything useful when the verifier reads
the same evidence the generator did.

The repository contains the data pipeline for that study. It prepares a
HotpotQA subset, builds a clean retrieval corpus, and generates controlled
poisoned versions of that corpus so that generator and verifier behaviour can
be compared under identical questions with and without misleading evidence.

## Research Question

How does the degree of independence between the generator and verifier affect
their tendency to fail together under knowledge poisoning?

## Why This Matters

If a poisoned passage misleads the generator, and the verifier is handed the
same passage, the verifier has nothing independent to check against. Swapping
in a different model family may not help, because both models are reasoning
over the same bad evidence. What may matter more is whether the verifier
retrieves its own evidence, through its own retriever, over its own index.

Verifier-based defences are being deployed on the assumption that a second
model adds a real safety layer. If that independence is superficial, so is the
protection.

## Dataset

HotpotQA, distractor configuration.

- 7,405 dev questions, each with 10 context paragraphs: 2 gold supporting
  passages and 8 distractors
- All questions in the dev split are labelled `hard`
- 5,918 bridge-type and 1,487 comparison-type questions

The dev split is used as the question pool. Questions are split 80/20 into
development (5,924) and test (1,481) sets at the question level under a fixed
seed. The resulting ID lists are committed to this repository. The split was
fixed before any attack was tuned, and the test set is reserved for checking
whether results hold on questions the pipeline was never adapted to.

Source data is not redistributed. `src/data/load_hotpotqa.py` downloads it.

- Source: https://huggingface.co/datasets/hotpotqa/hotpot_qa
- Paper: Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop
  Question Answering*, EMNLP 2018
- License: CC BY-SA 4.0. Derived artifacts here inherit its share-alike terms.

## Poisoning Method

**Corpus.** Passages are flattened out of their per-question bundles into a
single global pool of 66,581 unique passages, deduplicated by title. Retrieval
runs over this whole pool rather than each question's own ten passages. In a
per-question pool a poisoned passage would be retrieved by construction and
attack success would measure nothing; against 66,581 competitors it has to
actually outrank the alternatives.

**Gold evidence is preserved.** Poisoned passages are added alongside the
original supporting passages, never in place of them. The true and false claims
coexist, and retrieval decides which one the generator sees. Removing gold
would guarantee attack success by deleting the truth rather than competing with
it, and would make clean and poisoned conditions incomparable.

**Poison is unmarked.** A poisoned passage carries the title of the passage it
was derived from and is otherwise indistinguishable. Nothing in the corpus flags
it. Provenance is written to a separate file that never enters the corpus, so
the retriever, generator, and verifier stay blind to which passages are
poisoned. That file is read only when scoring.

**Three copies per question.** Each poisoned claim is inserted three times with
distinct passage IDs. This is partly to give poison a realistic chance of
reaching top-k, and partly because one of the planned experiments asks whether
repeated copies of a single false claim are mistaken for independent
corroboration.

### Factual substitution

The implemented attack. It finds the value the question actually asks for and
changes only that.

The rule is narrow on purpose: a number in a passage is poisoned only if that
same number appears in the question's gold answer. So for a question whose
answer is `3,677 seated`, the script locates `3,677` in the gold passage and
replaces it with a different plausible figure, leaving every other number in
the passage untouched.

An earlier version picked the first number it found in the gold sentence. That
poisoned 7,352 questions, but most of those substitutions changed values the
question did not depend on — a director's birth year altered on a question
about his nationality, for example. The passage became false without becoming
misleading. Anchoring on the answer cut coverage to 1,099 questions and made
every poisoned passage attack the fact the answer rests on.

Coverage is 1,099 of 7,405 questions, or 14.8%, because most HotpotQA answers
are entities or yes/no rather than numerals. Of those, 1,063 are bridge-type
and 36 are comparison-type; comparison questions rarely qualify because their
answers are usually yes/no or a name rather than a quantity.

The answer field itself is never modified. It is read only to locate the target
and remains the ground truth used for scoring.

### Validation

Substitutions are matched on whole tokens, so `90` cannot match inside `1990`
or `3,907`. Every occurrence of the target value in a passage is replaced
rather than just the first, so a passage stating the same figure twice cannot
end up contradicting itself. The replacement is guaranteed to differ from the
original, so no question can be recorded as poisoned while its text is
unchanged.

The merge step asserts that no poisoned passage ID collides with a clean one,
that no passage ID is duplicated, and that the merged corpus size equals the sum
of its parts. Any failure halts the run rather than writing a corpus that would
be trusted downstream.

### Contradiction: what was tried and why it was set aside

Contradictory-passage poisoning was attempted with rule-based substitution.
Four approaches were built and evaluated, and each failed for a different
reason. The sequence is recorded here because it shapes how the remaining
attack types will be built.

**Substitute a random answer from the dataset.** For each question, the answer
was located in the gold passage and replaced with another question's answer.
This covered 4,194 questions, but replacements ignored what kind of thing the
answer was. A location was replaced by a number, producing text such as
"...based in 16,825...". Grammatically broken rather than factually misleading.

**Match on surface form.** Answers were bucketed by shape — year, numeric,
multi-word, single-word — and replacements drawn from the matching bucket. This
fixed the grammar, but "multi-word" held 4,851 answers of every possible type,
so a shopping mall's floor area could still be replaced by a person's name.
Three of three random samples were incoherent.

**Infer the type from the question.** Patterns over interrogatives — who,
where, how large — were used to decide what kind of answer was expected.
HotpotQA questions are multi-hop and name several entity types at once, so
"What government position was held by the woman who..." matched the person
pattern even though the answer is a job title. Roughly 38% of questions still
landed in an untyped pool, and mislabelled cases contaminated the pools they
were sampled from.
