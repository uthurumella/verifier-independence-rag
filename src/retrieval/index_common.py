"""Shared pieces of the retrieval pipeline: config, corpus loading, encoders, indexes.

Index layout, one folder per corpus under index_root:

    <index_root>/<variant>/
        passage_ids.json        row i of every index below is this passage
        dense/index.faiss       exact inner-product index (IndexFlatIP)
        dense/meta.json         locked settings, model commit, input hashes
        dense/poison_embeddings.npy   poisoned variants only; reused when unchanged
        bm25/                   bm25s index files
        bm25/meta.json

Row order is the clean passages in file order, then the variant's poisoned
passages in file order. That is the same order build_poisoned_corpus.py writes
corpus.jsonl in, so row numbers line up with that file too.

Later tasks load indexes through DenseIndex.load and BM25Index.load.
"""

import hashlib
import json
import os

import numpy as np
import yaml

DEFAULT_CONFIG = "configs/index_config.yaml"


# ---------------------------------------------------------------- config / io

def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def title_passage_id(title):
    """Same ID scheme as src/data/build_clean_corpus.py (clean passages only)."""
    return "p_" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- corpora

def clean_passages_path(cfg):
    return cfg["paths"]["clean_passages"]


def poison_passages_path(cfg, variant):
    return os.path.join(cfg["paths"]["poisoned_root"], variant, "poisoned_passages.jsonl")


def provenance_path(cfg, variant):
    return os.path.join(cfg["paths"]["poisoned_root"], variant, "provenance.jsonl")


def format_passage(cfg, passage):
    return cfg["passage_template"].format(title=passage["title"], text=passage["text"])


def load_corpus(cfg, variant):
    """Passages of one corpus in index row order: clean first, then poison."""
    passages = read_jsonl(clean_passages_path(cfg))
    if variant != cfg["clean_variant"]:
        poison = read_jsonl(poison_passages_path(cfg, variant))
        clean_ids = {p["passage_id"] for p in passages}
        poison_ids = [p["passage_id"] for p in poison]
        assert not clean_ids & set(poison_ids), f"{variant}: poisoned passage_id collides with a clean one"
        assert len(set(poison_ids)) == len(poison_ids), f"{variant}: duplicate poisoned passage_id"
        passages = passages + poison
    return passages


def index_dir(cfg, variant):
    return os.path.join(cfg["paths"]["index_root"], variant)


# ---------------------------------------------------------------- locked settings

def dense_lock(cfg, revision):
    d = cfg["dense"]
    return {
        "passage_template": cfg["passage_template"],
        "model_name": d["model_name"],
        "revision": revision,
        "query_prefix": d["query_prefix"],
        "passage_prefix": d["passage_prefix"],
        "max_seq_length": d["max_seq_length"],
        "normalize": d["normalize"],
    }


def bm25_lock(cfg):
    b = cfg["bm25"]
    return {
        "passage_template": cfg["passage_template"],
        "method": b["method"],
        "k1": b["k1"],
        "b": b["b"],
        "stopwords": b["stopwords"],
        "stemmer": b["stemmer"],
    }


# ---------------------------------------------------------------- dense encoder

class DenseEncoder:
    """Sentence-transformers encoder with the locked E5 prefixes applied."""

    def __init__(self, cfg, revision):
        import torch
        from sentence_transformers import SentenceTransformer

        d = cfg["dense"]
        self.cfg = cfg
        self.lock = dense_lock(cfg, revision)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(d["model_name"], revision=revision, device=self.device)
        self.model.max_seq_length = d["max_seq_length"]
        if d.get("fp16") and self.device == "cuda":
            self.model.half()
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts, show_progress):
        d = self.cfg["dense"]
        emb = self.model.encode(
            texts,
            batch_size=d["batch_size"],
            normalize_embeddings=d["normalize"],
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def encode_passages(self, passages, show_progress=True):
        prefix = self.cfg["dense"]["passage_prefix"]
        return self._encode([prefix + format_passage(self.cfg, p) for p in passages], show_progress)

    def encode_queries(self, queries, show_progress=False):
        prefix = self.cfg["dense"]["query_prefix"]
        return self._encode([prefix + q for q in queries], show_progress)


def resolve_revision(cfg):
    """Model commit to use: config value, else the one the clean index was built with, else latest."""
    if cfg["dense"].get("revision"):
        return cfg["dense"]["revision"]
    clean_meta = os.path.join(index_dir(cfg, cfg["clean_variant"]), "dense", "meta.json")
    if os.path.exists(clean_meta):
        return read_json(clean_meta)["lock"]["revision"]
    from huggingface_hub import model_info
    return model_info(cfg["dense"]["model_name"]).sha


def make_encoder(cfg, revision=None):
    return DenseEncoder(cfg, revision or resolve_revision(cfg))


# ---------------------------------------------------------------- loaders for later tasks

class DenseIndex:
    def __init__(self, index, passage_ids, meta):
        self.index = index
        self.passage_ids = passage_ids
        self.meta = meta

    @classmethod
    def load(cls, cfg, variant):
        import faiss

        d = index_dir(cfg, variant)
        index = faiss.read_index(os.path.join(d, "dense", "index.faiss"))
        ids = read_json(os.path.join(d, "passage_ids.json"))
        meta = read_json(os.path.join(d, "dense", "meta.json"))
        assert index.ntotal == len(ids), f"{variant}: dense index rows != passage ids"
        return cls(index, ids, meta)

    def search_vectors(self, query_vecs, k):
        scores, rows = self.index.search(query_vecs, k)
        return [[self.passage_ids[r] for r in row if r >= 0] for row in rows], scores

    def search(self, encoder, queries, k):
        assert encoder_matches(encoder, self.meta), "encoder settings differ from the ones this index was built with"
        return self.search_vectors(encoder.encode_queries(queries), k)


def encoder_matches(encoder, meta):
    return encoder.lock == meta["lock"]


class BM25Index:
    def __init__(self, retriever, passage_ids, cfg):
        self.retriever = retriever
        self.passage_ids = passage_ids
        self.cfg = cfg

    @classmethod
    def load(cls, cfg, variant):
        import bm25s

        d = index_dir(cfg, variant)
        retriever = bm25s.BM25.load(os.path.join(d, "bm25"))
        ids = read_json(os.path.join(d, "passage_ids.json"))
        meta = read_json(os.path.join(d, "bm25", "meta.json"))
        assert meta["lock"] == bm25_lock(cfg), f"{variant}: BM25 index was built with different settings"
        return cls(retriever, ids, cfg)

    def search(self, queries, k):
        tokens = bm25_tokenize(self.cfg, queries, show_progress=False)
        rows, scores = self.retriever.retrieve(tokens, k=min(k, len(self.passage_ids)), show_progress=False)
        return [[self.passage_ids[r] for r in row] for row in rows], scores


def bm25_tokenize(cfg, texts, show_progress=True):
    """Tokens as strings, so queries map onto the index vocabulary by word."""
    import bm25s

    b = cfg["bm25"]
    stemmer = None
    if b.get("stemmer"):
        import Stemmer
        stemmer = Stemmer.Stemmer(b["stemmer"])
    return bm25s.tokenize(
        texts,
        lower=True,
        stopwords=b["stopwords"],
        stemmer=stemmer,
        return_ids=False,
        show_progress=show_progress,
    )
