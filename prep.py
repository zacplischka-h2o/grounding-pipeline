"""Build the grounding corpus from RAGTruth. See docs/adr/0006-*.md.

Also owns the prompt and the serializer, because train.py and evaluate.py must use
byte-identical text and there is no shared module (four-file rule).

    python prep.py
"""

import json
import random
import subprocess
from pathlib import Path

REPO = "https://github.com/ParticleMedia/RAGTruth.git"
RAW = Path("data/raw/RAGTruth")
OUT = Path("data")
DEV_SOURCES = 160  # sources held out of train for dev; ~800 Records at 6 per source
SEED = 17

JUDGE_PROMPT = """You check whether an assistant's answer is grounded in its evidence.

The evidence is everything the assistant was given.

<evidence>
%s
</evidence>

<answer>
%s
</answer>

The answer is grounded only if every claim in it is supported by the evidence.
Every number in the answer must appear in the evidence. A made-up figure, a
made-up fact or reason, or a claim the evidence does not state makes the answer
ungrounded.

Is the answer grounded or ungrounded? Reply with exactly one word:
grounded or ungrounded."""


def render(record):
    """Record -> the exact prompt string the model sees. The only renderer.

    Evidence is a dict for Data2txt and QA, but a plain string for Summary. Passing a
    string through json.dumps would deliver a quote-wrapped, backslash-escaped
    one-liner unlike anything the model trained on.
    """
    ev = record["evidence"]
    ev = ev if isinstance(ev, str) else json.dumps(ev, indent=2)
    return JUDGE_PROMPT % (ev, record["response"])


def fetch():
    if RAW.exists():
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(RAW)], check=True)


def read_ragtruth():
    d = RAW / "dataset"
    sources = {s["source_id"]: s for s in map(json.loads, open(d / "source_info.jsonl"))}
    for r in map(json.loads, open(d / "response.jsonl")):
        if r["quality"] != "good":
            continue
        src = sources[r["source_id"]]
        yield {
            "record_id": str(r["id"]),
            "evidence": src["source_info"],
            "response": r["response"],
            # ADR 0002: anything not in the evidence is ungrounded. An empty span
            # list is the annotators saying they found nothing invented.
            "label": "ungrounded" if r["labels"] else "grounded",
            "meta": {
                "task_type": src["task_type"],
                "model": r["model"],
                "source_id": r["source_id"],
                "ragtruth_split": r["split"],
            },
        }


def write(name, rows):
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_ug = sum(1 for r in rows if r["label"] == "ungrounded")
    print(f"  {name+'.jsonl':16s} {len(rows):5d} rows  ungrounded {n_ug/len(rows):.3f}")


def main():
    fetch()
    rows = list(read_ragtruth())

    d2t = [r for r in rows if r["meta"]["task_type"] == "Data2txt"]
    test = [r for r in d2t if r["meta"]["ragtruth_split"] == "test"]
    pool = [r for r in d2t if r["meta"]["ragtruth_split"] == "train"]

    # Group by source_id so no business appears on both sides of the dev boundary.
    sources = sorted({r["meta"]["source_id"] for r in pool})
    random.Random(SEED).shuffle(sources)
    dev_sources = set(sources[:DEV_SOURCES])
    dev = [r for r in pool if r["meta"]["source_id"] in dev_sources]
    train = [r for r in pool if r["meta"]["source_id"] not in dev_sources]

    # Transfer check (ADR 0006): the other two task types, never trained or tuned on.
    transfer = [
        r for r in rows
        if r["meta"]["task_type"] in ("QA", "Summary")
        and r["meta"]["ragtruth_split"] == "test"
    ]

    print("built:")
    for name, rs in [("train", train), ("dev", dev), ("test", test), ("transfer", transfer)]:
        write(name, rs)

    assert not ({r["meta"]["source_id"] for r in train} & dev_sources), "dev leaked into train"
    assert not ({r["meta"]["source_id"] for r in train} &
                {r["meta"]["source_id"] for r in test}), "test leaked into train"

    # The prompt is load-bearing; show it once so a wrong render is visible immediately.
    print("\nprompt sample (first 400 chars of a dev record):")
    print(render(dev[0])[:400])


if __name__ == "__main__":
    main()
