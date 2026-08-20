# A script-only baseline is the first row in the table, and it runs before any training

The negatives in this corpus are dominated by the `minimal-edit` channel, and most minimal
edits change a number. ADR 0002's grounding rule — every number in the Response must appear
in the Evidence — is therefore a string-membership test that plain code can run exactly, at
zero cost, with no model at all. A prior prototype already had this as its `tracer`.

**Correction (2026-08-20):** this ADR originally claimed the tracer was never run as a
detector. That is false — the prototype's `composed` metric is exactly script-OR-model, so
every `composed` figure already includes it. Worse, measurement shows the script is not a
weak floor: with absolute-value normalisation it flags **0 of 408** grounded dev rows and
**0 of 269** grounded test rows at recall ~0.42, and the labelling rule *is* this rule on
the numeric subset with zero exceptions across 677 grounded rows. See
[the style-separability note](../research/2026-08-20-the-corpus-is-style-separable.md).

So the final table has three rows, not two: **script only**, Gemma off-the-shelf, Gemma
fine-tuned. The script row is produced **before any training happens**.

Two reasons this ordering is load-bearing:

1. **It is the real floor.** If a thirty-line number-matching script recovers most of the
   recall, then a large share of any reported fine-tuning lift is string matching rather
   than groundedness, and we want that known before a day is spent training.
2. **It measures the open question instead of guessing at it.** The gap between the script
   and the model is precisely the capability in doubt — whether the model can tell "this
   number came from the licensing field" from "this number appears somewhere in the blob".

Rejected alternative: teaching the model to emit the supporting field path alongside its
verdict. RSAT (arXiv 2605.00199) reports that retrofitted attribution collapses below 13%
format success, and that its working version needed a GRPO stage with an NLI faithfulness
reward — far outside the scope of this experiment. The verdict stays one word.

Documented fallback if Gemma 4 E2B will not train: a small encoder with a classification
head. `answerdotai/ModernBERT-large` is 395M, Apache 2.0, and covers the p99 4,095-token
input in one 8,192-token pass with no truncation; LettuceDetect built this exact detector
and reports 79.22 example-level F1 on RAGTruth against GPT-4's 63.4. This is recorded so
the option is not rediscovered from scratch — it is not the current path, because ADR 0001
locks the base model and the experiment's question is what fine-tuning does to *that* model.
