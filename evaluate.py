"""Score candidates on dev and test, freeze the threshold on dev, report.

    python evaluate.py script          # number-membership floor, no model  (ADR 0005)
    python evaluate.py answer-only     # response text only, no evidence    (ADR 0006)
    python evaluate.py gemma           # off-the-shelf Gemma 4 E2B
    python evaluate.py gemma-ft        # + LoRA adapter
    python evaluate.py report          # print the table from saved scores

Every candidate emits one score in [0,1] per Record: P(ungrounded). Thresholds are
tuned on dev at FPR <= 5% and applied unchanged to test, so the realized test FPR is
reported next to recall — a dev-frozen threshold does not hold its FPR.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

DATA = Path("data")
SCORES = Path("data/scores")
TARGET_FPR = 0.05

# ---------------------------------------------------------------- data


def load(split):
    return [json.loads(l) for l in open(DATA / f"{split}.jsonl")]


def y(rows):
    return np.array([r["label"] == "ungrounded" for r in rows], dtype=int)


# ---------------------------------------------------------------- candidates

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers(text):
    """Every number in a string, as absolute floats.

    Absolute value matters: evidence stores debits as -18.74 and a response quotes
    them as $18.74. Signed matching flags ~9% of grounded rows as false positives.
    """
    out = set()
    for m in NUM.findall(text):
        try:
            out.add(abs(float(m.replace(",", ""))))
        except ValueError:
            pass
    return out


def score_script(train, rows):
    """Fraction of the response's numbers that are absent from the evidence."""
    del train
    scores = []
    for r in rows:
        have = numbers(json.dumps(r["evidence"]))
        want = numbers(r["response"])
        scores.append(0.0 if not want else len(want - have) / len(want))
    return np.array(scores)


def score_answer_only(train, rows):
    """Bag of words over the response alone. The evidence is never seen.

    This is the floor the classifier must beat (ADR 0006). Digits are stripped so a
    high score cannot be explained by number checking.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    strip = lambda t: re.sub(r"[\d$%.,]", " ", t)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    x = vec.fit_transform([strip(r["response"]) for r in train])
    model = LogisticRegression(max_iter=3000).fit(x, y(train))
    return model.predict_proba(vec.transform([strip(r["response"]) for r in rows]))[:, 1]


def score_gemma(train, rows, adapter=None):
    """P(ungrounded) from the first generated token. See train.py for the recipe."""
    del train
    from prep import render
    from train import first_token_scores

    return first_token_scores([render(r) for r in rows], adapter=adapter)


CANDIDATES = {
    "script": score_script,
    "answer-only": score_answer_only,
    "gemma": score_gemma,
    "gemma-ft": lambda tr, rows: score_gemma(tr, rows, adapter="models/gemma-ft"),
}

# ---------------------------------------------------------------- metrics


def freeze_threshold(scores, labels, target=TARGET_FPR):
    """Lowest threshold whose dev FPR stays within target. Maximises recall there."""
    fpr, _, thr = roc_curve(labels, scores)
    ok = thr[fpr <= target]
    return float(ok[-1]) if len(ok) else float(thr[0])


def measure(scores, labels, threshold):
    flag = scores >= threshold
    pos, neg = labels == 1, labels == 0
    return {
        "n": int(len(labels)),
        "auroc": float(roc_auc_score(labels, scores)),
        "recall": float(flag[pos].mean()),
        "fpr": float(flag[neg].mean()),
    }


def run(name):
    train = load("train")
    out = {"threshold": None, "splits": {}, "by_model": {}}
    dev, test = load("dev"), load("test")

    dev_s = CANDIDATES[name](train, dev)
    out["threshold"] = freeze_threshold(dev_s, y(dev))
    out["splits"]["dev"] = measure(dev_s, y(dev), out["threshold"])

    for split, rows in [("test", test), ("transfer", load("transfer"))]:
        s = CANDIDATES[name](train, rows)
        out["splits"][split] = measure(s, y(rows), out["threshold"])
        if split == "test":
            for m in sorted({r["meta"]["model"] for r in rows}):
                idx = [i for i, r in enumerate(rows) if r["meta"]["model"] == m]
                out["by_model"][m] = measure(s[idx], y(rows)[idx], out["threshold"])

    SCORES.mkdir(parents=True, exist_ok=True)
    (SCORES / f"{name}.json").write_text(json.dumps(out, indent=2))
    print(f"{name}: threshold {out['threshold']:.4f}")
    for split, m in out["splits"].items():
        print(f"  {split:9s} n={m['n']:5d}  AUROC {m['auroc']:.4f}  "
              f"recall {m['recall']:.4f}  FPR {m['fpr']:.4f}")
    return out


def report():
    rows = []
    for name in CANDIDATES:
        f = SCORES / f"{name}.json"
        if f.exists():
            rows.append((name, json.loads(f.read_text())))
    if not rows:
        sys.exit("nothing scored yet")

    print("\n| Candidate | test AUROC | test recall @ dev FPR<=5% | realized test FPR | transfer AUROC |")
    print("|---|---|---|---|---|")
    for name, d in rows:
        t, x = d["splits"]["test"], d["splits"].get("transfer", {})
        print(f"| `{name}` | {t['auroc']:.3f} | {t['recall']:.3f} | {t['fpr']:.3f} | "
              f"{x.get('auroc', float('nan')):.3f} |")

    print("\nPer writer model, test AUROC:\n")
    models = sorted(rows[0][1]["by_model"])
    print("| Candidate | " + " | ".join(m.replace("-0613", "") for m in models) + " |")
    print("|---" * (len(models) + 1) + "|")
    for name, d in rows:
        print(f"| `{name}` | " + " | ".join(f"{d['by_model'][m]['auroc']:.3f}" for m in models) + " |")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "report"
    report() if arg == "report" else run(arg)
