# RAGTruth Data2txt replaces the synthetic corpus

The synthetic corpus is retired. A response-text-only classifier reaches test AUROC 0.989
on it with every digit deleted — higher than the prior fine-tuned model's bare AUROC — and
no subset survives filtering (see
[the style-separability note](../research/2026-08-20-the-corpus-is-style-separable.md)).
It cannot measure grounding, so it cannot answer this repo's question.

**The corpus is now RAGTruth** (https://github.com/ParticleMedia/RAGTruth, MIT), task type
`Data2txt`, `quality == "good"`: 6,195 Records over 1,033 sources. The Evidence is a nested
JSON business record — fields, nested objects, and genuine `null` values — which is the
closest public analogue to a tool response. The Responses are real generations from six
LLMs, and humans annotated the invented spans. A Record is **grounded** when its span list
is empty and **ungrounded** otherwise, which matches ADR 0002's rule: anything not in the
Evidence is ungrounded.

Measured on the same probe that killed the synthetic corpus, response-only with digits
stripped: **AUROC 0.837, recall 0.316 at FPR <= 5%** across all six writer models, and
**0.593-0.773** within a single writer model. Real, but not a giveaway.

## What changes

- **The answer-only probe is a permanent table row, not a pass/fail gate.** It is the floor
  the Classifier must beat, and it is reported next to every result.
- **Results are broken down per writer model.** Ungrounded rates range from 0.263
  (gpt-3.5-turbo) to 0.952 (llama-2-13b), so writer identity partly predicts the label.
  A pooled number hides that; six rows do not.
- **The transfer check is now RAGTruth's other two task types** — `QA` and `Summary` — held
  out entirely from training and tuning, read once. This is free, it comes from the same
  human annotators, and it isolates the variable that matters: does the capability survive
  a change of evidence shape.
- **Contrast-consistency and the per-negative-channel split from ADR 0004 no longer apply.**
  RAGTruth has no minted negatives and no contrast pairs; its ungrounded Records are natural
  model hallucinations. The remaining checks from ADR 0004 stand.
- **The script-only baseline of ADR 0005 stands** and is still row one: the Evidence is
  numeric JSON, so number membership is still a meaningful floor.
- **Splits**: RAGTruth's own `split` field gives the test set (900 Records). Dev is carved
  from its train split, grouped by `source_id` so no source appears on both sides.

## What is given up

The banking domain. The Evidence is business and review data, not accounts and
transactions, so no number here transfers to Commonwealth Bank as an accuracy estimate.
That is the accepted price: this repo answers whether fine-tuning improves grounding
accuracy, and it now answers that on data whose labels no generator of ours produced.

The synthetic corpus stays retired until its Responses are regenerated with style
alignment and re-cleared by the probe.
