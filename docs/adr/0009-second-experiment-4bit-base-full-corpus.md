# 9. Second experiment: 4-bit base, full training corpus

Date: 2026-08-21

## Status

Accepted. Extends [0003](0003-train-on-unsloth-not-local-mlx.md); does not supersede
[0008](0008-the-bars-are-set-against-the-writer-prior.md) — the first experiment and
its bars stand as reported.

## Context

The first experiment answered its question: LoRA on 1,000 rows lifts Gemma 4 E2B from
below every shortcut to statistically indistinguishable from Claude Opus 5 on the
per-writer mean. All four bars passed. The stop rule in `PLAN.md` closed that run.

`report.md` names one limitation as the most important: **the fine-tune is strong
where hallucination is blatant and weak where it is subtle.** On gpt-4-written
Responses — the writer that most resembles a careful production agent — `gemma-ft`
scores 0.594, barely above chance, where the Judge scores 0.895.

The 1,000-row sample gave the model roughly 324 gpt-written rows, of which about 90
were ungrounded. The full training pool holds 1,446 gpt-written rows, 409 of them
ungrounded — 4.5x the exposure to exactly the case it fails.

Separately, the intended consumer is a Tier 1 gate in another repo, where the
3.3 GB 4-bit checkpoint matters more than the 5.6 GB bf16 one.

## Decision

Run one more training run: **4-bit base, all 4,335 training rows.** Everything else —
prompt, renderer, LoRA rank, learning rate, epochs, splits, seeds — is unchanged.

Two variables move at once. That is deliberate and it is a real limit: this run
cannot attribute a change to quantization or to corpus size separately. It is not
trying to. The question is whether a 4-bit, full-corpus Classifier is usable, and
the run is scored so the fine-tuning effect stays clean — `gemma-4bit` (the same
base, no adapter) is scored alongside it, so the primary delta is measured within
one quantization.

The bf16 adapter is **not** overwritten. It saves to `models/gemma-ft-4bit`, and the
new eval rows are `gemma-4bit` / `gemma-ft-4bit`, so both experiments stay in the
report side by side.

### Pre-registered questions, written before the run

| # | Question | Passes if |
|---|---|---|
| 1 | Does the extra data fix the subtle case? | test AUROC on gpt-4-written Records rises above **0.594**, the bf16 run's figure, and the 95% CI on the paired delta excludes 0 |
| 2 | Does the fine-tune still beat its own base? | ΔAUROC(`gemma-ft-4bit` − `gemma-4bit`) on test, 95% CI excludes 0 |
| 3 | Is it still above the shortcut floor? | mean per-writer test AUROC > **0.611** |
| 4 | Does it hold on an unseen task shape? | transfer AUROC > **0.695** (the `writer-prior` figure) |

Question 1 is the reason this run exists. Questions 2-4 are the first experiment's
bars restated, so a regression cannot hide behind a win on 1.

## Consequences

The stop rule still binds. **One run.** A missed question is reported as a missed
question, not retuned. A miss on question 1 is itself a finding, and a useful one:
it says the subtle case is not a data-volume problem, and that a small Classifier
belongs at Tier 1 as a blatant-error filter with the Judge behind it — a different
architecture, not a failure.

Two eval rows are added to the report rather than replacing the existing ones. Six
candidate rows become eight. The four-file limit on pipeline code is unaffected.

Quantization cost is not free and is now folded into the measured number rather than
argued about: if `gemma-4bit` scores below `gemma`, that is visible in the table.
