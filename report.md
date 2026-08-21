# Does fine-tuning improve grounding accuracy?

**Yes. All four pre-registered bars pass.** LoRA on 1,000 rows moved a 2.3B Gemma 4 E2B
from below every shortcut to statistically indistinguishable from Claude Opus 5 on the
measure that shortcuts cannot fake.

| | off the shelf | fine-tuned | change |
|---|---|---|---|
| test AUROC | 0.706 | **0.863** | **+0.157** (CI +0.124–+0.189) |
| per-writer mean (shortcut-proof) | 0.586 | **0.733** | **+0.147** (CI +0.086–+0.207) |
| transfer AUROC (unseen task shape) | 0.680 | **0.782** | **+0.102** (CI +0.078–+0.129) |

---

## Read this first — what this number is not

- **Not banking.** The Evidence is public business and review data, not accounts and
  transactions. No number here is an accuracy estimate for a banking gate.
- **Factual statements only.** Production grades three statement classes; this corpus
  contains only the first ([ADR 0007](docs/adr/0007-this-repo-measures-factual-statements-only.md)).
- **Prevalence is inverted.** This corpus is 64% ungrounded; production is almost certainly
  mostly grounded. AUROC and recall-at-fixed-FPR carry over. **Precision does not.**
- **The Judge is a strong reference, not your production gate.** Claude Opus 5 on the same
  prompt, not whatever model Commonwealth Bank runs.
- **n = 900 test**, n ~ 150 per writer model. A single per-writer cell carries a 95%
  interval of roughly +/-0.11; the six-cell mean roughly +/-0.05.
- **The fine-tune is weak exactly where production is hard.** See
  [the caveat that matters](#the-caveat-that-matters). It is the most important limitation
  in this report.

The one decision this licenses: **LoRA fine-tuning is worth doing for this task.**

---

## Corpus

RAGTruth `Data2txt` ([ADR 0006](docs/adr/0006-ragtruth-data2txt-replaces-the-synthetic-corpus.md)),
MIT. Evidence is a nested JSON business record with real `null` values; Responses are real
generations from six LLMs; labels are human span annotations across four types — Evident
Conflict (4,169 spans), Evident Baseless Info (3,616), Subtle Baseless Info (1,434), Subtle
Conflict (63). That taxonomy is the same one `CONTEXT.md` states as "unsupported,
contradicted, or absent", arrived at independently.

Train 4,335 / dev 960 / test 900, grouped by `source_id`. Transfer: RAGTruth `QA` +
`Summary` test rows (1,775), never trained or tuned on.

**Training**: 1,000 rows sampled with a fixed seed — 701 ungrounded / 299 grounded, 574
distinct businesses, zero overlap with test. LoRA r=8, alpha=8, all-linear, lr 2e-4 cosine,
2 epochs, bf16 on an A100. 9.5 minutes. 18.5M trainable parameters, 0.36% of the model.

---

## The bars, written down before the model ran

| # | Bar | Result | |
|---|---|---|---|
| 1 | dAUROC(`gemma-ft` - best model-free row) on test, 95% CI excludes 0 | +0.027, CI **+0.002–+0.054** | **PASS**, narrowly |
| 2 | Mean per-writer-model test AUROC > 0.611 | **0.733** | **PASS** |
| 3 | Transfer AUROC > 0.695, and above every model-free row on both task types | **0.782**; QA 0.849, Summary 0.768 | **PASS** |
| 4 | dAUROC(`gemma-ft` - `gemma`) on test, 95% CI excludes 0 | +0.157, CI **+0.124–+0.189** | **PASS** |

Bars were set against the **writer-prior** baseline, not invented
([ADR 0008](docs/adr/0008-the-bars-are-set-against-the-writer-prior.md)). None was moved
after seeing a result. One training run at 1,000 rows, as the stop rule required.

---

## Results

### Main

| Candidate | test AUROC | test AUROC 95% CI | test recall | test FPR | transfer AUROC | transfer recall | transfer FPR |
|---|---|---|---|---|---|---|---|
| `script` | 0.588 | 0.557–0.620 | 0.016 | 0.003 | 0.513 | 0.102 | 0.048 |
| `writer-prior` | 0.828 | 0.800–0.857 | 0.238 | 0.037 | 0.695 | 0.190 | 0.164 |
| `answer-only` | 0.835 | 0.803–0.866 | 0.358 | 0.059 | 0.522 | 0.000 | 0.000 |
| `judge` (Claude Opus 5) | 0.849 | 0.827–0.872 | 0.964 | 0.265 | 0.836 | 0.904 | 0.232 |
| `gemma` off the shelf | 0.706 | 0.670–0.743 | 0.250 | 0.103 | 0.680 | 0.297 | 0.097 |
| **`gemma-ft`** | **0.863** | **0.840–0.885** | 0.613 | 0.065 | **0.782** | 0.280 | 0.043 |

### Test AUROC by writer model — the shortcut-proof column

`writer-prior` scores exactly 0.500 here by construction: its score never varies within a
writer. Nothing in this column can be earned by recognising who wrote the text.

| Candidate | gpt-3.5-turbo | gpt-4 | llama-2-13b | llama-2-70b | llama-2-7b | mistral-7B | mean |
|---|---|---|---|---|---|---|---|
| `script` | 0.541 | 0.494 | 0.570 | 0.493 | 0.547 | 0.636 | **0.547** |
| `writer-prior` | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | **0.500** |
| `answer-only` | 0.561 | 0.562 | 0.660 | 0.565 | 0.630 | 0.689 | **0.611** |
| `judge` | 0.821 | 0.895 | 0.583 | 0.667 | 0.669 | 0.618 | **0.709** |
| `gemma` | 0.586 | 0.532 | 0.593 | 0.607 | 0.607 | 0.591 | **0.586** |
| **`gemma-ft`** | 0.593 | 0.594 | **0.857** | **0.799** | **0.800** | **0.753** | **0.733** |

### Transfer AUROC by task type — never trained on

| Candidate | QA | Summary |
|---|---|---|
| `script` | 0.497 | 0.526 |
| `writer-prior` | 0.669 | 0.718 |
| `answer-only` | 0.583 | 0.505 |
| `judge` | 0.849 | 0.823 |
| `gemma` | 0.778 | 0.591 |
| **`gemma-ft`** | **0.849** | **0.768** |

### Deltas — paired cluster bootstrap over `source_id`, 2,000 resamples

| Comparison | split | dAUROC | 95% CI | excludes 0 |
|---|---|---|---|---|
| `gemma-ft` - `answer-only` | test | +0.027 | +0.002–+0.054 | **yes** |
| `gemma-ft` - `gemma` | test | +0.157 | +0.124–+0.189 | **yes** |
| `gemma-ft` - `answer-only` | transfer | +0.260 | +0.222–+0.299 | **yes** |
| `gemma-ft` - `gemma` | transfer | +0.102 | +0.078–+0.129 | **yes** |

### Per-writer mean, bootstrapped within writer (4,000 resamples)

| Comparison | d | 95% CI | verdict |
|---|---|---|---|
| `gemma-ft` - `gemma` | +0.147 | +0.086–+0.207 | fine-tuned higher |
| `gemma-ft` - `answer-only` | +0.122 | +0.065–+0.177 | fine-tuned higher |
| `gemma-ft` - `judge` | +0.024 | -0.024–+0.072 | **not separable** |

---

## What the numbers say

**Fine-tuning worked, and the lift is real.** Every comparison is an interval, not a point
estimate, and every one excludes zero. On the pooled test column the gain over the best
shortcut is small (+0.027) — but pooled numbers are inflated by writer identity, which the
fine-tune does not exploit. On the shortcut-proof column the gain is +0.122, and on the
unseen task shape it is +0.260.

**A 2.3B model is now indistinguishable from Claude Opus 5** where cheating is impossible:
0.733 against 0.709, a difference whose interval straddles zero. On transfer QA the two are
identical at 0.849.

**Off the shelf, Gemma was worse than cheating.** 0.706 pooled — below both shortcuts — and
0.586 per writer, below the 0.611 shortcut floor. It answered `grounded` to nearly
everything: its dev threshold froze at 0.0000 and its test false-alarm rate blew past the 5%
target to 10.3%. Fine-tuning fixed the bias as well as the accuracy — `gemma-ft` lands at
6.5% FPR against a 5% target, the tightest of any model row.

**`script` is honest and useless here.** 0.588. RAGTruth's hallucinations are invented
facts, attributes and sentiment, not wrong figures; it abstains on 118 of 900 test rows for
lack of any number to check. A number-matching gate would catch almost nothing on this data.

**The Judge runs hot.** It flags 96.4% of ungrounded Responses and 26.5% of grounded ones.
`gemma-ft` sits at 61.3% and 6.5%. Those are different operating points, not different
quality — which is why AUROC is the comparison and recall is printed beside its realized FPR.

---

## The caveat that matters

**The fine-tune is strong where hallucination is blatant and weak where it is subtle — and
production is the subtle case.**

| Writer | How often it hallucinates | `judge` | `gemma-ft` |
|---|---|---|---|
| gpt-4 | 30% | **0.895** | 0.594 |
| gpt-3.5-turbo | 27% | **0.821** | 0.593 |
| llama-2-13b | 96% | 0.583 | **0.857** |
| llama-2-7b | 88% | 0.669 | **0.800** |
| llama-2-70b | 85% | 0.667 | **0.799** |
| mistral-7B | 93% | 0.618 | **0.753** |

The two are almost mirror images. Opus is strong on the careful writers and weak on the
reckless ones; `gemma-ft` is the reverse. Equal means hide completely different skills.

A production financial agent is a careful writer: errors are rare and subtle. That is the
gpt-4 column, where `gemma-ft` scores **0.594** — barely above chance — and Opus scores
0.895. So the equal-on-average result must not be read as "a small model can replace the
judge". On the traffic that resembles production, it currently cannot.

Two things follow. The complementarity is an opportunity — an ensemble, or routing
low-confidence cases to the Judge, would beat either alone. And the obvious next experiment
is to train on subtle cases specifically, rather than on a mix dominated by blatant ones:
701 of 1,000 training rows were ungrounded, most of them from the reckless writers.

---

## Method notes

- Every candidate emits one score in [0,1] per Record: P(ungrounded). Per-row scores are
  saved so intervals stay computable.
- Gemma's score is the first-token probability of `ungrounded` normalized against
  `grounded`. Neither word is a single token under Gemma 4 — `grounded` splits as
  `['ground','ed']` and `ungrounded` as `['ung','rounded']` — but their first tokens differ,
  which is what the readout needs. Space-prefixed spellings are excluded: `' un'` would
  collect `' unfortunately'`, `' unclear'`, `' unsupported'`.
- Scored one row at a time. Padding put the readout on a pad slot — measured
  P(grounded)+P(ungrounded) = 1.1e-07 batched versus 1.00 unbatched — and Gemma 4's ~262k
  vocabulary made batched logits a 12 GB tensor. An assert on the probability mass caught
  the padding bug before it could produce a full table of noise.
- One `MODEL_ID` (`unsloth/gemma-4-E2B-it`, bf16) for training and both eval rows. The LoRA
  adapter is attached, never merged — merging shifts the first-token ratio.
- Reproduce: `python prep.py`, then
  `python evaluate.py {script,writer-prior,answer-only,judge,gemma,gemma-ft}`, then
  `python evaluate.py report`.

## Prior work in this repo

The first corpus was synthetic and had to be abandoned: a bag of words over the Response
alone, with every digit deleted, scored **AUROC 0.989** on its test split
([measurement](docs/research/2026-08-20-the-corpus-is-style-separable.md)). The bars and
eight silent defects were found by two rounds of adversarial review
([ADR 0008](docs/adr/0008-the-bars-are-set-against-the-writer-prior.md)).
