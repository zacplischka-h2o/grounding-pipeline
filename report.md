# Does fine-tuning improve grounding accuracy?

One question: how much better is a fine-tuned Gemma 4 E2B than the same model off the
shelf, at deciding whether an answer is grounded in its evidence?

Corpus: RAGTruth `Data2txt` (see [ADR 0006](docs/adr/0006-ragtruth-data2txt-replaces-the-synthetic-corpus.md)).
Evidence is a nested JSON business record; Responses are real generations from six LLMs;
labels are human span annotations. Train 4,335 / dev 960 / test 900, grouped by
`source_id`. Transfer set: RAGTruth `QA` + `Summary` test rows (1,775), never trained or
tuned on.

Thresholds are frozen on dev at FPR <= 5% and applied unchanged to test, so the realized
test FPR is reported beside recall.

---

## The bars, written down before the model ran

Pre-registered so no outcome can be read as a pass after the fact.

| Bar | Value | Why |
|---|---|---|
| Beat the answer-only floor on test AUROC | **> 0.835** | Below this, the Classifier is worse than a bag of words that never sees the Evidence. |
| Beat the answer-only floor per writer model | **> 0.61 mean** | The pooled 0.835 is mostly "which model wrote this". Per-model is the honest read. |
| Show something on the transfer set | **> 0.55 AUROC** | The answer-only floor scores 0.522 there — chance. Any real capability must beat chance where the style shortcut does not reach. |
| Fine-tuned beats off-the-shelf | **McNemar p < 0.05** on paired test decisions | The lift must be larger than noise at n = 900. |

If the fine-tuned model clears the first three and misses the fourth, the honest reading is
"Gemma 4 E2B is already competent here and fine-tuning adds little" — a real answer.

---

## Results

### Main table

<!--MAIN-->
| Candidate | test AUROC | test recall @ dev FPR<=5% | realized test FPR | transfer AUROC |
|---|---|---|---|---|
| `script` | 0.524 | 0.126 | 0.047 | 0.596 |
| `answer-only` | 0.835 | 0.358 | 0.059 | 0.522 |
| `gemma` | _pending_ | | | |
| `gemma-ft` | _pending_ | | | |
<!--/MAIN-->

### Per writer model, test AUROC

<!--BYMODEL-->
| Candidate | gpt-3.5-turbo | gpt-4 | llama-2-13b | llama-2-70b | llama-2-7b | mistral-7B |
|---|---|---|---|---|---|---|
| `script` | 0.547 | 0.478 | 0.637 | 0.491 | 0.545 | 0.599 |
| `answer-only` | 0.561 | 0.562 | 0.660 | 0.565 | 0.630 | 0.689 |
<!--/BYMODEL-->

---

## What the two model-free rows already tell you

**`script` — number membership, no model.** Test AUROC 0.524, barely above chance. On the
old synthetic corpus this same idea was a perfect-precision detector, because that corpus's
labels *were* this rule. RAGTruth's hallucinations are mostly invented facts, attributes and
sentiment rather than wrong figures, so number checking cannot find them. Good: the floor is
low and honest, and there is real work left for a model to do.

**`answer-only` — bag of words over the Response, evidence blanked, digits stripped.**
Test AUROC 0.835. That looks high, but the per-model table shows why: within a single writer
model it drops to 0.561-0.689. The pooled figure is largely writer identity — the Llama
models hallucinate 84-95% of the time and the GPT models 26-28%, so recognising the writer
half-answers the question. This is why every result is reported per model.

**The transfer set is the clean measurement.** The answer-only floor scores **0.522** there —
chance. The style shortcut does not survive a change of evidence shape. So the transfer
column is the least corruptible number in this report, and it is the one to trust.

---

## Method notes

- Every candidate emits one score in [0,1] per Record: P(ungrounded).
- Gemma's score is the first-token probability of `ungrounded`, normalized against
  `grounded`. No sampling. The LoRA adapter is attached, never merged, because merging
  shifts that ratio.
- One `MODEL_ID` (`unsloth/gemma-4-E2B-it`, 4-bit) for training and for both eval rows.
- Reproduce: `python prep.py`, then `python evaluate.py {script,answer-only,gemma,gemma-ft}`,
  then `python evaluate.py report`.

## Prior work in this repo

The first corpus for this experiment was synthetic and had to be abandoned: a bag of words
over the Response alone, with every digit deleted, scored **AUROC 0.989** on its test split.
Full measurement in
[the style-separability note](docs/research/2026-08-20-the-corpus-is-style-separable.md).
