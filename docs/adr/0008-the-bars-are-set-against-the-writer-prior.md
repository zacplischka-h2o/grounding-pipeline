# The bars are set against the writer-prior, and every comparison is an interval

Three adversarial reviews of the first plan found that the pre-registered bars were
calibrated against the wrong reference, and that the one significance test measured the
wrong thing. All four bars were clearable by a model that never reads the Evidence.

## What was measured

A **writer-prior** baseline — six numbers, the per-writer ungrounded rate on train, no
text and no Evidence read at all — scores **test AUROC 0.828** and **transfer AUROC 0.695**.
The original bars were `test > 0.835` and `transfer > 0.55`. The transfer bar sat well
*below* a six-entry lookup table.

The cause is in the corpus: ungrounded rates run 0.271 (gpt-3.5-turbo) to 0.964
(llama-2-13b-chat), so recognising who wrote the Response half-answers the question.

Two more reference points, both below what a real Classifier must clear: response **word
count alone** scores 0.600 on transfer, and `script` scored 0.596 there before its
abstention bug was fixed.

## The bars, restated

| # | Bar | Reference it is set against |
|---|---|---|
| 1 | ΔAUROC(`gemma-ft` − best model-free row) on test, **95% CI excludes 0** | `answer-only`, test AUROC 0.835 (CI 0.803-0.866) |
| 2 | Mean per-writer-model test AUROC **> 0.611** | `answer-only`'s mean. `writer-prior` scores exactly 0.500 here by construction, so this column is the writer-identity-free measure. |
| 3 | Transfer AUROC **> 0.695**, and above every model-free row on **both** task types | `writer-prior`, transfer 0.695 |
| 4 | ΔAUROC(`gemma-ft` − `gemma`) on test, **95% CI excludes 0** | itself |

## Why McNemar was dropped

Bar 4 was McNemar on paired thresholded decisions. Each candidate freezes its **own** dev
threshold, so the two decision vectors sit at different operating points and McNemar
measures where the thresholds landed, not which model discriminates better. Demonstrated:
taking one fixed score vector and moving its threshold from FPR<=0.05 to FPR<=0.30 gives
McNemar p = 3.1e-34 — the same model, the same ranking, overwhelming "significance".

It is replaced by a **paired cluster bootstrap on ΔAUROC**, resampling `source_id` so
Records sharing a business move together. Threshold-free, so it cannot be gamed by an
operating point. Every headline number in `report.md` now carries a 95% interval; no bar is
a comparison of two point estimates.

## Also fixed in the same pass

- **`script` abstention.** A Response with no numbers scored 0.0 — ranked maximally
  grounded. Number-free Responses are *more* likely to be ungrounded than the base rate, so
  the convention did the separating. Now 0.5, carrying no rank information. Test AUROC
  moved 0.524 -> 0.588 and transfer 0.596 -> 0.513.
- **Per-row scores are saved.** Only aggregates were persisted, so no interval or
  matched-FPR comparison would have been computable once the GPU session ended.
- **Transfer is broken out by task type.** Half of it is `Summary`, whose Evidence is a
  plain string, not a dict; `render()` was passing it through `json.dumps`, delivering a
  quote-wrapped escaped one-liner unlike anything in training. Fixed, and the two task
  types are now reported separately.
- **Recall and FPR are printed for transfer.** The dev-frozen threshold fires on *nothing*
  there for `answer-only` (recall 0.000, FPR 0.000) — invisible when only AUROC was shown.
- **`SFTConfig(max_length=...)`.** TRL defaults to 1024 and truncates from the right, which
  would have cut the answer word — the only supervised token — off a large share of rows,
  training the adapter on nothing while the loss looked normal.
- **Single-token label assert.** If `ungrounded` tokenizes as `un` + `grounded`, its
  first-token mass absorbs "Unfortunately", "Unless", "Under". The old assert passed in
  exactly that case.
- **A dead readout now stops the run** instead of being rescued by `clamp()` into
  plausible noise.
- **Double `<bos>`** removed from both the training text and the eval prompt.
