"""
Pure-Python scorer for the Verifi misinformation detector.

Loads detector_model.json (produced by train_detector.py) and scores text with
the same TF-IDF + logistic regression maths scikit-learn uses, so the web server
needs no heavy dependencies (important on PythonAnywhere's free plan).
"""
import json
import math
import os
import re

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_model.json")
_TOKEN = re.compile(r"(?u)\b\w\w+\b")
MIN_WORDS = 8          # below this the model has too little text to say anything
_MODEL = None
_FILLER = set("the a an and or of to in on at for with by from that this these those is are was were be been "
              "it its as you your we our they their he she his her i not no but if so do does did can will "
              "would could should have has had than then there here what which who whom".split())


def _load():
    global _MODEL
    if _MODEL is None:
        with open(MODEL_PATH, "r", encoding="utf-8") as f:
            _MODEL = json.load(f)
    return _MODEL


def model_info():
    return _load()["meta"]


def _features(text, vocab, idf):
    tokens = _TOKEN.findall(text.lower())
    grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    counts = {}
    for g in grams:
        i = vocab.get(g)
        if i is not None:
            counts[i] = counts.get(i, 0) + 1
    vec = {i: (1.0 + math.log(c)) * idf[i] for i, c in counts.items()}   # sublinear tf * idf
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0:
        return {}, len(tokens)
    return {i: v / norm for i, v in vec.items()}, len(tokens)


def score_text(text):
    """Return probability (0..1) that the text resembles misinformation + the words that drove it."""
    m = _load()
    vec, n_tokens = _features(text, m["vocab"], m["idf"])
    if n_tokens < MIN_WORDS or not vec:
        return {"status": "insufficient", "n_words": n_tokens, "probability": None, "drivers": []}
    z = m["intercept"] + sum(m["coef"][i] * v for i, v in vec.items())
    p = 1.0 / (1.0 + math.exp(-z))
    inv = {i: t for t, i in m["vocab"].items()}
    contrib = sorted(((m["coef"][i] * v, inv[i]) for i, v in vec.items()), reverse=True)
    drivers = []
    for c, t in contrib:
        if c <= 0 or len(drivers) == 4:
            break
        if not all(w in _FILLER for w in t.split()):   # skip pure filler words like "the", "that"
            drivers.append(t)
    return {"status": "ok", "n_words": n_tokens, "probability": p, "drivers": drivers}
