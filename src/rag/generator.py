"""Prompt building and the instruction-tuned generator (Hugging Face transformers).

The generator only ever sees the question and the retrieved passages, never
which passages are poisoned. Decoding is greedy, so the same prompt always
gives the same answer on the same hardware and library versions.

A later verifier (Task 2/3) can reuse build_prompt-style formatting and the
exact passage lists saved in predictions.jsonl, so "same evidence" really is
the same evidence.
"""

import hashlib
import json


def truncate_words(text, max_words):
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " ..."


def format_passages(cfg, passages):
    p = cfg["prompt"]["passage"]
    n = cfg["generator"]["max_passage_words"]
    return "\n\n".join(
        p.format(i=i, title=x["title"], text=truncate_words(x["text"], n)) for i, x in enumerate(passages, start=1)
    )


def build_messages(cfg, question, passages):
    user = cfg["prompt"]["user"].format(passages=format_passages(cfg, passages), question=question)
    return [{"role": "system", "content": cfg["prompt"]["system"].strip()}, {"role": "user", "content": user}]


def prompt_hash(cfg):
    return hashlib.sha256(json.dumps(cfg["prompt"], sort_keys=True).encode("utf-8")).hexdigest()[:10]


def clean_answer(text):
    """First line of the reply, without an 'Answer:' label or trailing full stop."""
    text = text.strip().split("\n")[0].strip()
    for label in ("Answer:", "answer:", "A:"):
        if text.startswith(label):
            text = text[len(label):].strip()
    if text.endswith(".") and not text.endswith("..."):
        text = text[:-1]
    return text.strip().strip('"').strip()


def resolve_model_revision(model_name, revision):
    if revision:
        return revision
    from huggingface_hub import model_info
    return model_info(model_name).sha


class HFGenerator:
    def __init__(self, cfg, revision):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        g = cfg["generator"]
        self.cfg = cfg
        self.revision = revision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, g["dtype"]) if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(g["model_name"], revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(g["model_name"], revision=revision, torch_dtype=dtype)
        self.model.to(self.device).eval()
        self.torch = torch

    def prompt_text(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, list_of_messages):
        g = self.cfg["generator"]
        texts = [self.prompt_text(m) for m in list_of_messages]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=g["max_new_tokens"],
                do_sample=g["do_sample"],
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tok.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1]:]
        raw = self.tok.batch_decode(new, skip_special_tokens=True)
        n_in = enc["attention_mask"].sum(dim=1).tolist()
        return [{"raw": r, "answer": clean_answer(r), "prompt_tokens": int(n)} for r, n in zip(raw, n_in)]
