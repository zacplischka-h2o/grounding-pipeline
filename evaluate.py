"""Score candidates on dev and test, freeze the threshold on dev, report.

    python evaluate.py script          # number-membership floor, no model  (ADR 0005)
    python evaluate.py writer-prior    # per-writer base rate, no text      (ADR 0008)
    python evaluate.py answer-only     # response text only, no evidence    (ADR 0006)
    python evaluate.py judge           # the incumbent LLM judge, via the Anthropic API
    python evaluate.py gemma           # off-the-shelf Gemma 4 E2B
    python evaluate.py gemma-ft        # + LoRA adapter
    python evaluate.py report          # print the tables from saved scores

Every candidate emits one score in [0,1] per Record: P(ungrounded). Thresholds are
tuned on dev at FPR <= 5% and applied unchanged to test and transfer, so the realized
FPR is reported beside recall — a dev-frozen threshold does not hold its FPR, and on
transfer (prevalence 0.205 vs test 0.643) it can fire on nothing at all.

Per-row scores are saved. Everything downstream — bootstrap intervals, matched-FPR
comparisons — needs them, and the GPU session that produced them will be gone.
"""

import collections
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

DATA = Path("data")
SCORES = Path("data/scores")
TARGET_FPR = 0.05
JUDGE_MODEL = "claude-opus-5"
JUDGE_WORKERS = 12
BOOTSTRAP = 2000
SEED = 17

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
    them as $18.74. Signed matching flags a chunk of grounded rows as false positives.
    """
    out = set()
    for m in NUM.findall(text):
        try:
            out.add(abs(float(m.replace(",", ""))))
        except ValueError:
            pass
    return out


def score_script(train, rows):
    """Fraction of the response's numbers that are absent from the evidence.

    A response with no numbers scores 0.5 — abstain, carrying no rank information.
    Scoring it 0.0 would rank it as maximally grounded, and number-free responses are
    *more* likely to be ungrounded than the base rate, so that convention inflated
    test AUROC and inverted the transfer number.
    """
    del train
    scores, abstained = [], 0
    for r in rows:
        want = numbers(r["response"])
        if not want:
            scores.append(0.5)
            abstained += 1
            continue
        have = numbers(json.dumps(r["evidence"]))
        scores.append(len(want - have) / len(want))
    print(f"  script abstained on {abstained}/{len(rows)} rows with no numbers")
    return np.array(scores)


def score_writer_prior(train, rows):
    """The base rate of the model that wrote the response. Six numbers, no text.

    This exists because it is a strong baseline, not a weak one: writer identity
    predicts the label well enough to clear a badly-set bar (ADR 0008). Anything
    claiming to read evidence must beat it.
    """
    rate = collections.defaultdict(list)
    for r in train:
        rate[r["meta"]["model"]].append(r["label"] == "ungrounded")
    prior = {m: float(np.mean(v)) for m, v in rate.items()}
    overall = float(np.mean([r["label"] == "ungrounded" for r in train]))
    return np.array([prior.get(r["meta"]["model"], overall) for r in rows])


def score_answer_only(train, rows):
    """Bag of words over the response alone. The evidence is never seen.

    Digits are stripped so a high score cannot be explained by number checking.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    strip = lambda t: re.sub(r"[\d$%.,]", " ", t)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    x = vec.fit_transform([strip(r["response"]) for r in train])
    model = LogisticRegression(max_iter=3000).fit(x, y(train))
    return model.predict_proba(vec.transform([strip(r["response"]) for r in rows]))[:, 1]


def score_judge(train, rows):
    """The incumbent Judge: one API call per Record, one word back.

    Binary by nature — a real gate returns a Verdict, not a score. So its "AUROC" is
    balanced accuracy, it needs no dev threshold, and it sits in the table as a
    reference point rather than as a bar.
    """
    del train
    import os
    from concurrent.futures import ThreadPoolExecutor

    import anthropic
    from prep import render

    for line in open(".env"):
        k, _, v = line.strip().partition("=")
        if k == "ANTHROPIC_API_KEY":
            os.environ.setdefault(k, v.strip().strip("\"'"))
    client = anthropic.Anthropic()

    def one(record):
        for attempt in range(4):
            try:
                m = client.messages.create(
                    model=JUDGE_MODEL, max_tokens=2048,
                    messages=[{"role": "user", "content": render(record)}],
                )
                text = " ".join(b.text for b in m.content if b.type == "text").lower()
                # 'ungrounded' contains 'grounded', so test for it first.
                if "ungrounded" in text:
                    return 1.0
                if "grounded" in text:
                    return 0.0
                return float("nan")
            except anthropic.APIStatusError as e:
                if e.status_code < 500 and e.status_code != 429:
                    raise
            except anthropic.APIConnectionError:
                pass
            time.sleep(2 ** attempt)
        return float("nan")

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        scores = list(pool.map(one, rows))
    bad = sum(1 for v in scores if v != v)
    if bad:
        print(f"  judge: {bad}/{len(rows)} records gave no usable verdict; scored 0.5")
    return np.array([0.5 if v != v else v for v in scores])


def score_gemma(train, rows, adapter=None):
    """P(ungrounded) from the first generated token. See train.py for the recipe."""
    del train
    from prep import render
    from train import first_token_scores

    return first_token_scores([render(r) for r in rows], adapter=adapter)


# Candidates that emit a Verdict, not a score: threshold is 0.5 and dev is not scored.
BINARY = {"judge"}

CANDIDATES = {
    "script": score_script,
    "writer-prior": score_writer_prior,
    "answer-only": score_answer_only,
    "judge": score_judge,
    "gemma": score_gemma,
    "gemma-ft": lambda tr, rows: score_gemma(tr, rows, adapter="models/gemma-ft"),
}

# ---------------------------------------------------------------- metrics


def auroc(labels, scores):
    """NaN rather than a crash when a slice holds one class."""
    return float(roc_auc_score(labels, scores)) if 0 < labels.mean() < 1 else float("nan")


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
        "n_pos": int(pos.sum()),
        "auroc": auroc(labels, scores),
        "recall": float(flag[pos].mean()) if pos.any() else float("nan"),
        "fpr": float(flag[neg].mean()) if neg.any() else float("nan"),
    }


def boot_auroc_ci(labels, scores, groups):
    """Cluster bootstrap over source_id. Returns (lo, hi) at 95%."""
    rng = np.random.default_rng(SEED)
    by_group = collections.defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    keys = list(by_group)
    vals = []
    for _ in range(BOOTSTRAP):
        idx = np.concatenate([by_group[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        if 0 < labels[idx].mean() < 1:
            vals.append(roc_auc_score(labels[idx], scores[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def boot_delta_ci(labels, a, b, groups):
    """Paired cluster bootstrap on AUROC(b) - AUROC(a). Same resamples for both."""
    rng = np.random.default_rng(SEED)
    by_group = collections.defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    keys = list(by_group)
    vals = []
    for _ in range(BOOTSTRAP):
        idx = np.concatenate([by_group[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        if 0 < labels[idx].mean() < 1:
            vals.append(roc_auc_score(labels[idx], b[idx]) - roc_auc_score(labels[idx], a[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def run(name):
    train = load("train")
    out = {"threshold": None, "splits": {}, "by_model": {}, "by_task": {}, "scores": {}}
    dev, test = load("dev"), load("test")

    if name in BINARY:
        out["threshold"] = 0.5  # a Verdict has no threshold to tune
    else:
        dev_s = CANDIDATES[name](train, dev)
        out["threshold"] = freeze_threshold(dev_s, y(dev))
        out["splits"]["dev"] = measure(dev_s, y(dev), out["threshold"])
        out["scores"]["dev"] = dev_s.tolist()

    # The judge costs real money per Record, so transfer (1,775 rows, ~2x the test
    # split) is opt-in: JUDGE_TRANSFER=1. Every free candidate always scores both.
    splits = [("test", test)]
    if name not in BINARY or os.environ.get("JUDGE_TRANSFER"):
        splits.append(("transfer", load("transfer")))
    for split, rows in splits:
        s = CANDIDATES[name](train, rows)
        out["scores"][split] = s.tolist()
        out["splits"][split] = measure(s, y(rows), out["threshold"])
        key = "model" if split == "test" else "task_type"
        bucket = out["by_model"] if split == "test" else out["by_task"]
        for v in sorted({r["meta"][key] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["meta"][key] == v]
            bucket[v] = measure(s[idx], y(rows)[idx], out["threshold"])

    SCORES.mkdir(parents=True, exist_ok=True)
    (SCORES / f"{name}.json").write_text(json.dumps(out))
    print(f"{name}: threshold {out['threshold']:.4f}")
    for split, m in out["splits"].items():
        print(f"  {split:9s} n={m['n']:5d}  AUROC {m['auroc']:.4f}  "
              f"recall {m['recall']:.4f}  FPR {m['fpr']:.4f}")
    return out


# ---------------------------------------------------------------- report


def report():
    saved = [(n, json.loads((SCORES / f"{n}.json").read_text()))
             for n in CANDIDATES if (SCORES / f"{n}.json").exists()]
    if not saved:
        sys.exit("nothing scored yet")
    test, transfer = load("test"), load("transfer")
    yt, yx = y(test), y(transfer)
    src = [r["meta"]["source_id"] for r in test]
    nan = float("nan")

    def cell(d, split, field):
        return d["splits"].get(split, {}).get(field, nan)

    print("\n### Main\n")
    print("| Candidate | test AUROC | test AUROC 95% CI | test recall | test FPR | "
          "transfer AUROC | transfer recall | transfer FPR |")
    print("|---|---|---|---|---|---|---|---|")
    for n, d in saved:
        lo, hi = boot_auroc_ci(yt, np.array(d["scores"]["test"]), src)
        print(f"| `{n}` | {cell(d,'test','auroc'):.3f} | {lo:.3f}–{hi:.3f} | "
              f"{cell(d,'test','recall'):.3f} | {cell(d,'test','fpr'):.3f} | "
              f"{cell(d,'transfer','auroc'):.3f} | {cell(d,'transfer','recall'):.3f} | "
              f"{cell(d,'transfer','fpr'):.3f} |")

    models = sorted({m for _, d in saved for m in d["by_model"]})
    print("\n### Test AUROC by writer model (n≈150 each; 95% CI is roughly ±0.11)\n")
    print("| Candidate | " + " | ".join(m.replace("-0613", "") for m in models) + " | mean |")
    print("|---" * (len(models) + 2) + "|")
    for n, d in saved:
        vals = [d["by_model"].get(m, {}).get("auroc", nan) for m in models]
        print(f"| `{n}` | " + " | ".join(f"{v:.3f}" for v in vals) +
              f" | **{np.nanmean(vals):.3f}** |")

    tasks = sorted({t for _, d in saved for t in d["by_task"]})
    if tasks:
        print("\n### Transfer AUROC by task type\n")
        print("| Candidate | " + " | ".join(tasks) + " |")
        print("|---" * (len(tasks) + 1) + "|")
        for n, d in saved:
            print(f"| `{n}` | " + " | ".join(
                f"{d['by_task'].get(t, {}).get('auroc', nan):.3f}" for t in tasks) + " |")

    # The bars: every comparison is a paired interval, never two point estimates.
    have = dict(saved)
    floor = max(("answer-only", "writer-prior"),
                key=lambda k: have[k]["splits"]["test"]["auroc"] if k in have else -1)
    print(f"\n### Deltas (paired cluster bootstrap, {BOOTSTRAP} resamples)\n")
    print("| Comparison | split | ΔAUROC | 95% CI | excludes 0 |")
    print("|---|---|---|---|---|")
    pairs = [(floor, "gemma-ft", "test"), ("gemma", "gemma-ft", "test"),
             (floor, "gemma-ft", "transfer"), ("gemma", "gemma-ft", "transfer")]
    for a, b, split in pairs:
        if a not in have or b not in have:
            continue
        labels, groups = (yt, src) if split == "test" else (yx, list(range(len(transfer))))
        sa, sb = np.array(have[a]["scores"][split]), np.array(have[b]["scores"][split])
        lo, hi = boot_delta_ci(labels, sa, sb, groups)
        d = roc_auc_score(labels, sb) - roc_auc_score(labels, sa)
        print(f"| `{b}` − `{a}` | {split} | {d:+.3f} | {lo:+.3f}–{hi:+.3f} | "
              f"{'**yes**' if lo > 0 else 'no'} |")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "report"
    report() if arg == "report" else run(arg)
