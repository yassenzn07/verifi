"""
Verifi - train the misinformation detector.

Trains a small TF-IDF + Logistic Regression classifier on CoAID
(COVID-19 Healthcare Misinformation Dataset, Cui & Lee, 2020,
https://github.com/cuilimeng/CoAID) and exports the weights as plain JSON
so the Flask app can score text in pure Python (no sklearn/torch needed on
the web server).

Run on any machine with: pip install scikit-learn pandas
    python train_detector.py
Output: detector_model.json (+ metrics printed to the console)
"""
import json
import random
import re
import urllib.request

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold

BASE = "https://raw.githubusercontent.com/cuilimeng/CoAID/master"
FOLDERS = ["05-01-2020", "07-01-2020", "09-01-2020", "11-01-2020"]
KINDS = {  # file stem -> label (1 = misinformation, 0 = reliable)
    "ClaimFakeCOVID-19": 1,
    "NewsFakeCOVID-19": 1,
    "ClaimRealCOVID-19": 0,
    "NewsRealCOVID-19": 0,
}
SEED = 7
MAX_FEATURES = 10000


def fetch(folder, stem):
    url = f"{BASE}/{folder}/{stem}.csv"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            raw = r.read()
    except Exception:
        return None
    if raw.startswith(b"404"):
        return None
    from io import BytesIO
    return pd.read_csv(BytesIO(raw), encoding="utf-8-sig")


def clean(t):
    t = str(t).replace("“", '"').replace("”", '"')
    t = re.sub(r"\s+", " ", t).strip().strip('"').strip()
    return t


def build_samples():
    texts, labels, groups, kinds = [], [], [], []
    gid = 0
    for folder in FOLDERS:
        for stem, label in KINDS.items():
            df = fetch(folder, stem)
            if df is None or "title" not in df.columns:
                continue
            for _, row in df.iterrows():
                units = []
                if isinstance(row.get("title"), str):
                    units.append(("claim" if stem.startswith("Claim") else "title", clean(row["title"])))
                if stem.startswith("News") and isinstance(row.get("content"), str):
                    units.append(("content", clean(row["content"])))
                for kind, text in units:
                    if len(text.split()) >= 4:
                        texts.append(text)
                        labels.append(label)
                        groups.append(gid)  # same article -> same group (no leakage)
                        kinds.append(kind)
                gid += 1
    df = pd.DataFrame({"text": texts, "label": labels, "group": groups, "kind": kinds})
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


def main():
    df = build_samples()
    print(f"samples: {len(df)}  misinformation: {int(df.label.sum())}  reliable: {int((1 - df.label).sum())}")
    print(df.groupby(["kind", "label"]).size().unstack(fill_value=0))

    # split by article group so title/content of one article never straddle train/test
    groups = sorted(df.group.unique())
    random.Random(SEED).shuffle(groups)
    test_groups = set(groups[: int(len(groups) * 0.2)])
    test = df[df.group.isin(test_groups)]
    train = df[~df.group.isin(test_groups)]

    def vec():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=MAX_FEATURES,
                               sublinear_tf=True, lowercase=True)

    # choose C by grouped CV on the training part only
    best_c, best_score = None, -1
    for c in (0.5, 2, 8, 32):
        scores = []
        for tr, va in GroupKFold(n_splits=5).split(train.text, train.label, train.group):
            v = vec()
            X = v.fit_transform(train.text.iloc[tr])
            m = LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(X, train.label.iloc[tr])
            p = m.predict(v.transform(train.text.iloc[va]))
            y = train.label.iloc[va]
            tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
            scores.append(2 * tp / max(2 * tp + fp + fn, 1))  # F1 of the misinformation class
        s = float(np.mean(scores))
        print(f"C={c}: CV F1(misinformation)={s:.3f}")
        if s > best_score:
            best_c, best_score = c, s

    v = vec()
    Xtr = v.fit_transform(train.text)
    model = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000).fit(Xtr, train.label)
    Xte = v.transform(test.text)
    pred = model.predict(Xte)
    report = classification_report(test.label, pred, target_names=["reliable", "misinformation"],
                                   output_dict=True, zero_division=0)
    print("\nHELD-OUT TEST SET")
    print(classification_report(test.label, pred, target_names=["reliable", "misinformation"], zero_division=0))
    print(confusion_matrix(test.label, pred))
    for kind in sorted(test.kind.unique()):
        mk = (test.kind == kind).values
        acc = float((pred[mk] == test.label.values[mk]).mean())
        print(f"  accuracy on '{kind}' samples: {acc:.3f} (n={int(mk.sum())}, misinformation n={int(test.label.values[mk].sum())})")

    # refit on everything for the shipped model
    v = vec()
    X = v.fit_transform(df.text)
    final = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000).fit(X, df.label)

    out = {
        "vocab": {t: int(i) for t, i in v.vocabulary_.items()},
        "idf": [float(x) for x in v.idf_],
        "coef": [float(x) for x in final.coef_[0]],
        "intercept": float(final.intercept_[0]),
        "meta": {
            "dataset": "CoAID (COVID-19 Healthcare Misinformation Dataset)",
            "n_samples": int(len(df)),
            "n_misinformation": int(df.label.sum()),
            "C": best_c,
            "heldout_accuracy": round(float((pred == test.label.values).mean()), 3),
            "heldout_f1_misinformation": round(report["misinformation"]["f1-score"], 3),
            "heldout_precision_misinformation": round(report["misinformation"]["precision"], 3),
            "heldout_recall_misinformation": round(report["misinformation"]["recall"], 3),
            "heldout_n": int(len(test)),
        },
    }
    with open("detector_model.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print("\nwrote detector_model.json", out["meta"])

    # sanity check: pure-python scorer must match sklearn exactly
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from detector import score_text
    import detector
    detector._MODEL = None
    diffs = []
    for t in test.text.head(400):
        r = score_text(t)
        if r["status"] != "ok":      # too short for the scorer's minimum-length rule
            continue
        diffs.append(abs(r["probability"] - final.predict_proba(v.transform([t]))[0, 1]))
    print(f"checked {len(diffs)} texts; max |pure-python - sklearn| probability diff:", max(diffs))


if __name__ == "__main__":
    main()
