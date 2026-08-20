# Plan

## The question

**How much does LoRA fine-tuning improve a small model's grounding accuracy?**

One question, one number, measured honestly. This is a personal experiment. Nothing is
handed to anyone. Latency and cost are explicitly out of scope — the assumption that a
2B model beats an 8-second API judge on speed is taken for granted, not measured.

## Definitions

From `CONTEXT.md`, which carries the production wording. A **Response** is **ungrounded**
if any factual component is unsupported, contradicted, or absent from the **Evidence**,
regardless of how plausible it seems. One such component condemns the whole Response. The
**Classifier** outputs a binary **Verdict**; the **Judge** is the incumbent LLM API gate it
would replace.

Production grades three statement classes at different strictness — factual statements,
capability claims, clarifying questions. **This repo measures class 1 only**
([ADR 0007](docs/adr/0007-this-repo-measures-factual-statements-only.md)).

## Corpus

RAGTruth `Data2txt`, `quality == "good"` ([ADR 0006](docs/adr/0006-ragtruth-data2txt-replaces-the-synthetic-corpus.md)).

| Split | Rows | Ungrounded | Source |
|---|---|---|---|
| train | 4,335 | 0.698 | RAGTruth train, minus the dev sources |
| dev | 960 | 0.673 | 160 sources held out of RAGTruth train, grouped by `source_id` |
| test | 900 | 0.643 | RAGTruth's own test split |
| transfer | 1,775 | 0.205 | RAGTruth `QA` + `Summary` test rows — never trained or tuned on |

Evidence is a nested JSON business record with real `null` values. Responses are real
generations from six LLMs. Labels are human span annotations: empty span list = grounded.

The first corpus for this experiment was synthetic and was retired. A bag of words over the
Response alone, digits deleted, scored **AUROC 0.989** on its test split
([measurement](docs/research/2026-08-20-the-corpus-is-style-separable.md)).

## Method

- **Base model**: `unsloth/gemma-4-E2B-it`, 4-bit, one `MODEL_ID` shared by training and
  both eval rows. Locked ([ADR 0001](docs/adr/0001-non-chinese-base-model.md)). No bake-off.
- **Training**: LoRA on a CUDA GPU via Unsloth, not local MLX
  ([ADR 0003](docs/adr/0003-train-on-unsloth-not-local-mlx.md)). `r=8`, `alpha=8`,
  `dropout=0`, `target_modules="all-linear"`, `lr=2e-4` cosine, 2 epochs, effective batch
  16, `max_seq_length=8192`, bf16 on an A100. 1,000 rows first; scale to 4,335 only if the lift is short.
- **Supervision**: completion-only, via Gemma 4's `<|turn>user\n` / `<|turn>model\n`
  markers. The supervised span is asserted to be the answer word plus terminator before
  training starts.
- **Readout**: P(`ungrounded`) at the first generated token, normalized against
  `grounded`. A continuous score, so thresholds move without re-running the model. The
  adapter is attached, never merged — merging shifts that ratio.
- **Thresholds**: frozen on dev at FPR <= 5%, applied unchanged to test and transfer. The
  realized test FPR is printed beside recall, because a dev-frozen threshold does not hold
  its FPR.
- **Test discipline**: dev is for tuning, test is read at the end.

## The table

Four candidates, all emitting one score in [0,1] per Record.

| Row | What it is | Why it is there |
|---|---|---|
| `writer-prior` | The writing model's base rate, no text | The identity floor ([ADR 0008](docs/adr/0008-the-bars-are-set-against-the-writer-prior.md)) |
| `script` | Number membership, no model | The deterministic floor ([ADR 0005](docs/adr/0005-script-only-baseline-runs-before-any-training.md)) |
| `answer-only` | Bag of words over the Response, Evidence blanked, digits stripped | The shortcut floor ([ADR 0006](docs/adr/0006-ragtruth-data2txt-replaces-the-synthetic-corpus.md)) |
| `gemma` | Off the shelf | The before |
| `gemma-ft` | + LoRA adapter | The after |

Columns: test AUROC, test recall at the dev-frozen threshold, realized test FPR, transfer
AUROC. Plus a per-writer-model breakdown, because pooled numbers are inflated by writer
identity.

## The bars

Set against the **writer-prior** baseline — six numbers, no text, no Evidence — which
scores test AUROC 0.828 and transfer 0.695. Any bar below that is clearable by a lookup
table. Stated in full in [`report.md`](report.md) and
[ADR 0008](docs/adr/0008-the-bars-are-set-against-the-writer-prior.md). Every comparison is
a paired cluster-bootstrap interval over `source_id`, never two point estimates.

**Stop rule: one training run at 1,000 rows. A missed bar is reported as a missed bar, not
retuned.** The prior prototype died of open-ended tuning; this is the line that prevents a
repeat.

## Order of work

1. `prep.py` — fetch RAGTruth, build the four splits. **Done.**
2. `evaluate.py script` — the deterministic floor. **Done: test AUROC 0.588.**
3. `evaluate.py writer-prior` — the identity floor. **Done: test AUROC 0.828, transfer 0.695.**
4. `evaluate.py answer-only` — the shortcut floor. **Done: test AUROC 0.835, transfer 0.522.**
5. `evaluate.py gemma` on 20 dev rows as a go/no-go, before renting a GPU for the full run.
6. `train.py` — LoRA, 1,000 rows, on Colab. **Pending.**
7. `evaluate.py gemma` and `evaluate.py gemma-ft`, full splits. **Pending.**
8. `evaluate.py report` → paste into `report.md`. **Pending.**

## Files

Four, hard limit. `prep.py`, `train.py`, `evaluate.py`, `report.md`. `prep.py` owns the
prompt and the renderer because train and eval must produce byte-identical text and there
is no shared module.

## Known limits

Stated in [`report.md`](report.md), in the deliverable rather than the plan.

## What would make this wrong

- The supervised span assert fires on the longest row → the sequence cap is truncating the answer word.
- The readout assert fires → the prompt truncated, or the model is not answering in one word.
- `gemma-ft` beats `gemma` on test but not on transfer → the lift is corpus-specific.
- `gemma-ft` fails to beat 0.611 per writer model → it learned writer identity, not grounding.
