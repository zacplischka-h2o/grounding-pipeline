# The corpus is style-separable — measured 2026-08-20

**ADR 0004's stop condition fired before any training happened.**

## What was run

A TF-IDF (1-2 gram) + logistic regression classifier, trained on the **response text alone**
from the 7,423 non-derivation rows of `data/train_v1/sft_train.jsonl` in the prior
prototype, evaluated on the held-out 510-row `data/v1/test.jsonl`.

No evidence. No model. No tool responses. Just the words of the answer.

| Input | AUROC | Recall @ FPR <= 5% |
|---|---|---|
| Response text as-is | **0.9889** | 0.9668 |
| Response text, every digit and `$ % . ,` stripped | **0.9894** | 0.9668 |

Reproduce: `scratchpad/probe.py`, run under the prototype's `.venv` (sklearn 1.6.1).
Splits verified disjoint by `world_seed`, so this is not leakage — it is register.

## Why this kills the measurement

The prior fine-tuned Gemma reported a **bare** AUROC of 0.988 on dev. A bag of words that
has never seen the evidence, and cannot see a single number, matches it on test.

So the reported lift is not evidence of grounding capability. The generator writes grounded
answers and ungrounded answers in measurably different prose registers, and that register
is the easiest signal in the data. Any model trained on this corpus will find it first.

This is the exact failure Xie et al. (KDD 2024, https://arxiv.org/abs/2410.12278) name:
*"any salient distinctions in language styles like length of text or tone between
hallucinated output and non-hallucinated output can be exploited as shortcuts during
supervised training."* Their fix is Language Style Alignment during generation.

## Three corroborating defects found in the same review

1. **The number-matching script is not a weak floor — it is a perfect-precision detector.**
   With absolute-value normalisation (debits are stored as `-18.74` and quoted as `$18.74`),
   it flags **0 of 408** grounded dev rows and **0 of 269** grounded test rows, at recall
   ~0.42. The labelling rule *is* this rule on the numeric subset, with zero exceptions
   across 677 grounded rows. So a large share of any "lift" is a model learning one `in`
   operator. **ADR 0005's premise that the tracer was never run as a detector is false** —
   the prototype's `composed` metric is exactly script-OR-model.
2. **~23% of test labels are not programmatic.** 116 of 510 rows carry
   `label.provenance == "llm-jury"`, mapping exactly onto the `organic` negatives, and 45
   rows (8.8%) have jury votes contradicting the assigned label. On that slice the
   experiment measures agreement with an LLM judge — the circularity the project exists to
   remove.
3. **The prior comparison numbers are composed and are dev, not test.** Bare off-the-shelf
   recall is 0.109, not 0.356; bare fine-tuned is 0.939, not 0.955
   (`eval/results/frozen/gemma-4-e2b*.json`, under the `"dev"` key).

## Execution defects that apply on any path forward

Recorded here so they are not rediscovered:

- **Rebuild the prompt from the prototype's code, do not rewrite it.** Train rows are
  pre-serialized strings; eval rows are raw. `record["response"]` is a **dict**
  (`{"turn":…, "text":…}`) — writing it into a prompt embeds `{'turn': 6, ...}` and nothing
  errors. Copy `eval/serialize.py`'s longctx branch and `JUDGE_PROMPT` verbatim, then assert
  a rebuilt row equals the stored `prompt`.
- **One `MODEL_ID` for training and eval, prequantized 4-bit, adapter never merged.** The
  prototype records that dequantize/merge shifts the first-token logit ratio.
- **Gemma 4's completion-only markers are `<|turn>user\n` / `<|turn>model\n`**, not Gemma 3's
  `<start_of_turn>`. A wrong marker masks everything or nothing, and the loss cannot tell
  you — Unsloth documents that a loss of 13-15 is normal for E2B. Decode row 0's unmasked
  labels and assert they are the answer word plus terminator.
- **`max_seq_length = 8192`, not 4096.** ~1% of dev/test exceeds 4,096 tokens; truncation
  cuts the trailing question, so `P(grounded)+P(ungrounded) ~ 0` and the readout silently
  returns 0.5.
- **`sft_train_1k.jsonl` already contains 154 derivation rows.** Filter, then sample.
- **The script row has no AUROC** (it emits a binary verdict), and a dev-frozen threshold
  does not hold its FPR on test — measured spread 1.9%-7.1%. Label the column "test recall
  at the dev-frozen threshold" and print the realized FPR beside it.
- **RAGTruth is a day of work, not a row.** No loader exists; it needs a join of
  `response.jsonl` and `source_info.jsonl` on `source_id`, a `task_type == Data2txt` filter,
  and a quality filter. Subsample to ~800 rows with a fixed seed before scoring.
- **Free Colab is a T4.** The prototype's reference numbers are A100 (1k-row train in 37
  minutes). Budget roughly 5x, on a VM with a ~4-hour session cap.

## The salvage attempt failed too

The obvious cheap fix is to keep only the `minimal-edit` contrast pairs — a negative and the
verified-grounded answer it was edited from, which ought to be style-matched by
construction. It is not:

| Slice | n | AUROC | Recall @ FPR <= 5% |
|---|---|---|---|
| All test rows, digits stripped | 510 | 0.9894 | 0.9668 |
| `organic` universe, digits stripped | 385 | **1.0000** | **1.0000** |
| `minimal-edit` pair universe, digits stripped | 250 | 0.9828 | 0.9440 |
| `minimal-edit` pair universe, digits kept | 250 | 0.9829 | 0.9440 |

The `organic` negatives are **perfectly** separable from prose register alone — they were
generated by a different path and read differently, completely. And the minimal edits leave
their own lexical fingerprint: 0.983 even with every digit deleted, on pairs that differ by
a few characters.

There is no subset of this corpus that measures grounding. Filtering does not fix it;
the responses have to be regenerated with style alignment, or the corpus has to be replaced.
