# Does fine-tuning improve grounding accuracy?

One question: how much better is a fine-tuned Gemma 4 E2B than the same model off the
shelf, at deciding whether an answer is grounded in its evidence?

**Status: the two model-free floors are measured. The model rows are pending a GPU run.**

---

## Read this first — what this number is not

- **Not banking.** The Evidence is public business and review data, not accounts and
  transactions. No number here transfers to a banking gate as an accuracy estimate.
- **Factual statements only.** Production grades three statement classes; this corpus
  contains only the first. Nothing here says whether a Classifier can catch an agent
  promising a transfer its tools cannot make
  ([ADR 0007](docs/adr/0007-this-repo-measures-factual-statements-only.md)).
- **Prevalence is inverted.** This corpus is 64% ungrounded; production is almost certainly
  mostly grounded. AUROC and recall-at-fixed-FPR carry over. **Precision does not.**
- **No Judge row.** The incumbent LLM judge is not scored on these rows, so this report
  cannot say whether a small model is good enough to replace it. It says only what
  fine-tuning adds.
- **n = 900 test**, and n ≈ 150 per writer model — a single per-writer cell carries a 95%
  interval of roughly ±0.11.

The one decision this report licenses: **whether LoRA fine-tuning is worth doing at all for
this task.** Nothing more.

---

## Corpus

RAGTruth `Data2txt` ([ADR 0006](docs/adr/0006-ragtruth-data2txt-replaces-the-synthetic-corpus.md)).
Evidence is a nested JSON business record; Responses are real generations from six LLMs;
labels are human span annotations. Train 4,335 / dev 960 / test 900, grouped by `source_id`.
Transfer: RAGTruth `QA` + `Summary` test rows (1,775), never trained or tuned on.

Thresholds are frozen on dev at FPR <= 5% and applied unchanged, so the realized FPR is
printed beside recall. Every headline number carries a 95% interval from a cluster
bootstrap over `source_id`.

---

## The bars, written down before the model ran

Set against the **writer-prior** — six numbers, the per-writer ungrounded rate, no text and
no Evidence read at all. It scores test AUROC 0.828 and transfer 0.695, so any bar below
that is clearable by a lookup table
([ADR 0008](docs/adr/0008-the-bars-are-set-against-the-writer-prior.md)).

| # | Bar | Set against |
|---|---|---|
| 1 | ΔAUROC(`gemma-ft` − best model-free row) on test, **95% CI excludes 0** | `answer-only`, 0.835 (CI 0.803–0.866) |
| 2 | Mean per-writer-model test AUROC **> 0.611** | `answer-only`'s mean; `writer-prior` scores 0.500 here by construction |
| 3 | Transfer AUROC **> 0.695**, and above every model-free row on **both** task types | `writer-prior`, 0.695 |
| 4 | ΔAUROC(`gemma-ft` − `gemma`) on test, **95% CI excludes 0** | itself |

**One training run at 1,000 rows. A missed bar is reported as a missed bar, not retuned.**
Clearing 1–3 but missing 4 reads as "Gemma 4 E2B is already competent here and fine-tuning
adds little" — a real answer.

---

## Results

### Main

| Candidate | test AUROC | test AUROC 95% CI | test recall | test FPR | transfer AUROC | transfer recall | transfer FPR |
|---|---|---|---|---|---|---|---|
| `script` | 0.588 | 0.557–0.620 | 0.016 | 0.003 | 0.513 | 0.102 | 0.048 |
| `writer-prior` | 0.828 | 0.800–0.857 | 0.238 | 0.037 | 0.695 | 0.190 | 0.164 |
| `answer-only` | 0.835 | 0.803–0.866 | 0.358 | 0.059 | 0.522 | 0.000 | 0.000 |
| `gemma` | _pending_ | | | | | | |
| `gemma-ft` | _pending_ | | | | | | |

### Test AUROC by writer model (n ≈ 150 each; 95% CI ≈ ±0.11)

| Candidate | gpt-3.5-turbo | gpt-4 | llama-2-13b | llama-2-70b | llama-2-7b | mistral-7B | mean |
|---|---|---|---|---|---|---|---|
| `script` | 0.541 | 0.494 | 0.570 | 0.493 | 0.547 | 0.636 | **0.547** |
| `writer-prior` | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | **0.500** |
| `answer-only` | 0.561 | 0.562 | 0.660 | 0.565 | 0.630 | 0.689 | **0.611** |

### Transfer AUROC by task type

| Candidate | QA | Summary |
|---|---|---|
| `script` | 0.497 | 0.526 |
| `writer-prior` | 0.669 | 0.718 |
| `answer-only` | 0.583 | 0.505 |

---

## What the model-free rows already tell you

**`writer-prior` is the finding.** Six numbers — the ungrounded rate of each writer model,
0.271 for gpt-3.5-turbo up to 0.964 for llama-2-13b — reach test AUROC **0.828**. It reads
no text and no Evidence. Any pooled number near 0.83 on this corpus means nothing on its
own, and the first version of this report had bars set *below* it.

**The per-writer-model column is the honest one.** `writer-prior` scores exactly 0.500
there by construction, because its score is constant within a writer. So that column is
free of writer identity, and `answer-only`'s mean of 0.611 is the real shortcut floor.

**`script` — number membership, no model.** Test AUROC 0.588. RAGTruth's hallucinations are
mostly invented facts, attributes and sentiment rather than wrong figures, so number
checking cannot find them. It abstains (0.5) on 118 of 900 test rows that contain no number.
Good: the floor is low and honest, and there is real work left for a model.

**`answer-only` fires on nothing in transfer.** Recall 0.000 at FPR 0.000. The dev-frozen
threshold does not survive the prevalence drop from 0.643 to 0.205. This is why recall and
FPR are printed for transfer and not just AUROC.

---

## Method notes

- Every candidate emits one score in [0,1] per Record: P(ungrounded). Per-row scores are
  saved, so intervals and matched-FPR comparisons stay computable after the GPU is gone.
- Gemma's score is the first-token probability of `ungrounded`, normalized against
  `grounded`. No sampling. The LoRA adapter is attached, never merged — merging shifts that
  ratio. A dead two-way distribution stops the run rather than being rescued into noise.
- One `MODEL_ID` (`unsloth/gemma-4-E2B-it`, 4-bit) for training and for both eval rows.
- Reproduce: `python prep.py`, then
  `python evaluate.py {script,writer-prior,answer-only,gemma,gemma-ft}`, then
  `python evaluate.py report`.

## Prior work in this repo

The first corpus for this experiment was synthetic and had to be abandoned: a bag of words
over the Response alone, with every digit deleted, scored **AUROC 0.989** on its test split
([measurement](docs/research/2026-08-20-the-corpus-is-style-separable.md)). The bars and
several silent defects in this pipeline were found by adversarial review
([ADR 0008](docs/adr/0008-the-bars-are-set-against-the-writer-prior.md)).
