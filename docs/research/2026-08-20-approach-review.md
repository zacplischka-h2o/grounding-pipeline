# Approach review — 2026-08-20

Research pass on the plan settled in the grilling session: fine-tune Gemma 4 E2B (LoRA)
as a binary groundedness classifier, measure the lift over the off-the-shelf model.

All dates below are the observed date, 2026-08-20.

---

## TL;DR

**The plan is sound. Three changes.**

1. **Train on a rented/free GPU with Unsloth, not locally with MLX.** Three open `mlx-lm`
   bugs hit this exact config. Unsloth has a working Gemma 4 E2B notebook and it fits in
   8–10 GB. This contradicts the earlier "local" decision — see [Change 1](#change-1).
2. **Add three cheap controls before believing any number.** The prior run's shape
   (AUROC 0.988, recall 0.955) is exactly what a shortcut looks like. The controls cost
   one flag each in `evaluate.py`.
3. **Add RAGTruth as the transfer check.** Public, MIT, ~17.8k human-annotated real LLM
   responses, and its data-to-text split feeds **JSON business data** as evidence. This is
   the right-shaped external check that ConvFinQA was not.

Everything else — Gemma 4 E2B locked, accuracy only, copy the old data, two-row table,
four files — holds.

---

## Evidence status

- **Local**: prior prototype at `/Users/zacharyplischka/prototypes/grounding` (read-only:
  `CONTEXT.md`, ADR 0005, the Gemma readout, `maxlen.json`, dataset schemas and counts).
  Nothing was executed.
- **arXiv**: 5 real API queries across cs.CL, cs.AI, cs.LG, cs.IR; 23 papers returned,
  14 read in isolation (one agent per paper, no cross-talk).
- **Web**: three parallel research agents — off-the-shelf detectors and benchmarks;
  MLX/Unsloth tooling and model identity; evaluation methodology.
- **Community sources** (X, Reddit, YouTube): not used. Official docs, model cards,
  papers and leaderboards were sufficient.
- **Not verified**: any wall-clock training time on Apple Silicon; licences for
  HaluBench / FaithBench / HalluLens; freshness of the LLM-AggreFact leaderboard; whether
  an "M5 Pro" 24 GB config exists (Apple's own published test machine was a plain M5 24 GB).

---

## Model identity — confirmed real

`google/gemma-4-E2B-it` is a correct, current model ID. Gemma 4 was announced
**2026-04-02**: sizes E2B / E4B / 26B-A4B / 31B, **Apache 2.0**, E2B at 128K context,
~5.1B total and ~2.3B effective parameters (per-layer embeddings).
Source: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ and
https://huggingface.co/google/gemma-4-E2B-it

ADR 0001 holds. Note `google/gemma-3n-E2B-it` is licensed under the Gemma licence, not
Apache — Gemma 4 is strictly the better pick.

MLX 4-bit checkpoints that exist: `mlx-community/gemma-4-e2b-it-4bit` (most used),
`mlx-community/gemma-4-E2B-it-qat-4bit`, and `mlx-community/Gemma4-E2B-IT-Text-int4`
(text-only, the cleanest for this task).

---

## Change 1 — train on Unsloth, not local MLX {#change-1}

This reverses the earlier decision. The reason is tooling risk, not preference.

**Open `mlx-lm` issues that hit this exact configuration** (verified open 2026-08-20,
against `mlx-lm` 0.31.3, released 2026-04-22):

| Issue | What breaks |
|---|---|
| #1363 (PR, open) | Gemma 4's `per_layer_model_projection` sits outside `model.layers`, so default LoRA target discovery **misses the per-layer-embedding path entirely**. It is never trained. |
| #1355 (PR, open) | NaN gradients when `mask_prompt=True` on chat-format data. Completion-only loss is the plan. |
| #1172 (issue, open) | `mlx_lm.fuse` on a quantised base **silently drops the adapter**. |
| #1658 / #1659 | No gradient clipping and no non-finite guard in the LoRA trainer. |

Also: `--max-seq-length` defaults to **2048**, which would silently truncate the p99 rows
(measured p99 was 4,095 tokens on a comparable corpus).

**Unsloth, by contrast** (https://unsloth.ai/docs/models/gemma-4/train, observed
2026-08-20): `unsloth/gemma-4-E2B-it`, LoRA at **8–10 GB VRAM**, free Colab notebooks,
`train_on_completions` for response-only loss, and Gemma-4-specific fixes already landed —
notably that `use_cache=False` corrupts attention and produces garbage logits on E2B/E4B.
Recommended settings on that page: `r=8`, `lora_alpha=8` (alpha ≥ r), `lora_dropout=0`,
`target_modules="all-linear"`, lr `2e-4` (`2e-5` for long runs).

Unsloth's own MLX training support is listed as "coming this month" — i.e. not ready.

**Corroborating local evidence**: the prior project's spec committed to local MLX, and its
readout says the run actually happened on Colab A100. Local did not hold last time.

**When this is wrong**: if a Colab/rental GPU is unavailable or the data cannot leave the
machine. It is synthetic fake-bank data, so that is not a real constraint here. If it were,
go local and pin `--max-seq-length 4096`, use `completion` (not `chat`) dataset format,
check the loss is not NaN at iteration 1, and never merge the adapter.

---

## Change 2 — three controls before believing the number

The prior run reported AUROC 0.988 and recall 0.955 at 4.9% FPR on a dev split from the
same generator family as the training data. That is the classic shape of a model that
learned the *minting process*, not groundedness.

The literature is unambiguous that this happens:

- SNLI labels are readable from the hypothesis **alone** at ~67% — the generation
  procedure imprints label signal into one field.
  Gururangan et al., NAACL 2018 — https://aclanthology.org/N18-2017/
- "Any salient distinctions in language style like length of text or tone between
  hallucinated output and non-hallucinated output can be exploited as shortcuts during
  supervised training" — the paper that most closely matches this build.
  Xie et al., KDD 2024 — https://arxiv.org/abs/2410.12278
- Counterfactually-augmented data **can exacerbate** existing spurious correlations, and
  narrow perturbation diversity limits the benefit.
  Joshi & He, ACL 2022 — https://aclanthology.org/2022.acl-long.256/
- A failing partial-input probe does **not** prove the data is clean.
  Feng, Wallace & Boyd-Graber, ACL 2019 — https://aclanthology.org/P19-1554/

**The three controls, in priority order.** Each is a flag in `evaluate.py`, not new code.

1. **Answer-only baseline** — blank the evidence, score the same test set. High AUROC means
   the dataset is cheatable and the headline number is not about grounding. (Low AUROC is
   inconclusive, per Feng et al. — this is a one-way test.)
2. **Contrast-consistency** — the old records carry `meta.contrast_of`, so minimal-edit
   pairs are already identifiable. Score both members of a pair and report the percentage
   where **both** are correct. This will be well below row-level accuracy. It is the honest
   number.
   Gardner et al., Findings of EMNLP 2020 — https://aclanthology.org/2020.findings-emnlp.117/
3. **Per-channel split** — `meta.negative_channel` distinguishes `minimal-edit` from
   `organic`. Report two AUROC lines, not one. If organic negatives are much harder, the
   aggregate is being carried by the easy half.

Optional fourth, if any doubt survives: **random-label control** — retrain on shuffled
labels and report selectivity (real accuracy minus control accuracy). Anything above ~0.55
AUROC on shuffled labels means leakage or memorisation.
Hewitt & Liang, EMNLP 2019 — https://aclanthology.org/D19-1275/

---

## Change 3 — RAGTruth is the transfer check

ADR 0002 dropped ConvFinQA, correctly: it tests multi-step arithmetic, which this agent
never does. But dropping it leaves no external reference at all, and a synthetic-only
result is not believable on its own.

**RAGTruth** (https://github.com/ParticleMedia/RAGTruth, MIT; paper
https://arxiv.org/pdf/2401.00396) is the right shape:

- ~17,790 responses over 2,965 sources, from 6 different LLMs, **human**-annotated at span
  level — not programmatic labels from one generator.
- Its **data-to-text split feeds structured JSON business data** (Yelp-style fields) as the
  evidence. Roughly 17.7% of hallucination spans in that split involve null JSON values.
  That is the closest public analogue to grounding an answer in tool-response JSON.
- It has published example-level numbers to sit next to: LettuceDetect (ModernBERT-large,
  395M) reports **79.22 example-level F1**, versus GPT-4 at 63.4 and a fine-tuned
  Llama-2-13B at 78.7.

Use it exactly as ConvFinQA was meant to be used: **never train on it**, read it once at
the end, report it as a separate row. If the fine-tune's lift is real capability it will
show something on RAGTruth; if it is corpus-specific it will collapse. Either answer is
worth having.

**Bluntly**: no public benchmark exists for grounding an agent's final answer in
banking-style JSON tool responses. RAGTruth's data-to-text split is the nearest thing.

---

## Comparables — what already exists off the shelf

Two of the three best open detectors are unusable here.

| Detector | Size | Licence | Origin | Score | Usable? |
|---|---|---|---|---|---|
| Bespoke-MiniCheck-7B | 7B | CC BY-**NC** | US lab, but fine-tuned from `internlm2_5-7b-chat` (Shanghai AI Lab) | 77.4 avg, #1 on LLM-AggreFact | **No** — non-commercial *and* Chinese-derived base |
| Patronus Lynx 8B | 8B | CC-BY-**NC** | Patronus AI, US | HaluBench 82.9 (GPT-4o 86.5) | **No** — non-commercial |
| **Granite Guardian 3.3 8B** | 8B | **Apache 2.0** | IBM, US | LLM-AggreFact 0.761, RAGTruth 0.831, TRUE 0.777 | **Yes** — also scores function-calling hallucination |
| HHEM-2.1-Open | 0.1B | Apache 2.0 | Vectara, US | AggreFact 76.55 BA, RAGTruth-QA 74.28 BA | Yes |
| MiniCheck-Flan-T5-Large | 0.8B | MIT | UT Austin, US | 75.0 avg | Yes |
| LettuceDetect (ModernBERT-large) | 0.395B | MIT | KR Labs et al. | RAGTruth example-F1 79.22 | Yes |

Sources: https://llm-aggrefact.github.io/ ,
https://huggingface.co/ibm-granite/granite-guardian-3.3-8b ,
https://huggingface.co/vectara/hallucination_evaluation_model ,
https://huggingface.co/KRLabsOrg

**What transfers**: small detectors sit only ~2–4 points behind a frontier judge on prose.
A 0.8B Flan-T5 scores 75.0 where GPT-4o scores 75.9. The premise of this project — a small
model can do this job — is well supported.

**What should not be copied**: their training corpora and their evidence format. All of the
above are tuned on prose passages.

**The finding that matters most**: on tool and code evidence the gap *inverts*. In
"Beyond Document Grounding" (https://arxiv.org/html/2607.00895v1, 2026-07-01) zero-shot LLM
judges score ≤0.22 F1 on code-agent evidence while a tuned 2B model reaches 0.602. A
non-Chinese 350M encoder (LFM2.5-Encoder-350M) reaches **0.854 example-level F1** — and
example-level F1 is exactly this project's binary gate metric. Structured evidence is where
fine-tuning earns its keep, which is the strongest argument yet for doing this experiment.

**No verified Gemma-based groundedness detector exists.** This would be the first.

---

## Papers read

Clustered by angle. Scores are relevance / practicality / rigour, 0–10.

### Cluster A — synthetic negative minting (**the chosen cluster**)

| Paper | Angle | Score |
|---|---|---|
| [2410.12278](https://arxiv.org/abs/2410.12278) Controlled Automatic Task-Specific Synthetic Data Generation for Hallucination Detection | Rank the real hallucination patterns, generate against them, align language style, filter out edits a surface heuristic already catches | `[rel9 prac8 rig8]` |
| [2407.05474](https://arxiv.org/abs/2407.05474) Perturbation-Based Synthetic Data Generation in System Responses | Rewrite responses into faithful *and* hallucinated variants; a T5-base beat zero-shot detectors on accuracy and latency | `[rel9 prac9 rig7]` |
| [2606.16307](https://arxiv.org/abs/2606.16307) State-Grounded Multi-Agent Synthetic Data Generation for Tool-Augmented LLMs | An authoritative world-state object makes every tool response derivable from state, killing tool-call fabrication by construction | `[rel7 prac6 rig6]` |

### Cluster B — shortcut / artifact risk

| Paper | Angle | Score |
|---|---|---|
| [2402.12715](https://arxiv.org/abs/2402.12715) The Clever Hans Mirage | Taxonomy of spurious-correlation failures and mitigations; worst-group evaluation | `[rel8 prac7 rig7]` |
| [2604.04518](https://arxiv.org/abs/2604.04518) Reproducibility study on finding spurious correlations | Correction methods compared under low data and severe imbalance; group labels are the bottleneck | `[rel7 prac6 rig8]` |
| [2511.07318](https://arxiv.org/abs/2511.07318) When Bias Pretends to Be Truth | Shortcut-driven hallucinations are confident, scale-immune, and evade confidence-based detection | `[rel6 prac5 rig8]` |

### Cluster C — evaluation discipline

| Paper | Angle | Score |
|---|---|---|
| [2606.06959](https://arxiv.org/abs/2606.06959) OpenHalDet | Freeze the whole eval chain — prompt, sampling, labels, scoring — so a measured difference is attributable | `[rel6 prac8 rig7]` |
| [2411.15594](https://arxiv.org/abs/2411.15594) A Survey on LLM-as-a-Judge | Judge reliability and bias taxonomy; how to audit the incumbent judge | `[rel5 prac6 rig6]` |
| [2404.07060](https://arxiv.org/abs/2404.07060) Groundedness in Retrieval-augmented Long-form Generation | Sentence-level support labels, OR-ed to a whole-answer verdict; correct-but-ungrounded is common | `[rel7 prac7 rig7]` |

### Cluster D — alternative detector mechanisms (not chosen)

| Paper | Angle | Score |
|---|---|---|
| [2605.00199](https://arxiv.org/abs/2605.00199) RSAT: Structured Attribution Makes SLMs Faithful Table Reasoners | SFT for a cited-JSON format, then GRPO against an NLI faithfulness reward | `[rel6 prac3 rig8]` |
| [2506.09886](https://arxiv.org/abs/2506.09886) Probabilistic distances-based hallucination detection with RAG | Unsupervised: distance between evidence-token and answer-token hidden-state distributions | `[rel4 prac7 rig6]` |
| [2410.11594](https://arxiv.org/abs/2410.11594) Black-box Uncertainty Quantification for LLM-as-a-Judge | Derive a high/low uncertainty label from token probabilities across cross-evaluated assessments | `[rel5 prac6 rig6]` |

---

## Prior-art pitfalls — watch-outs, not verdicts

- **The style tell.** Edited and unedited answers differ in length, tone, and digit
  density. That difference is learnable and is not grounding. (2410.12278)
- **Narrow edit vocabulary.** If minimal edits only ever touch a digit or a date, the model
  learns "scan the numbers" and ignores the rest of the answer. Audit edit-type coverage.
  (Joshi & He 2022; 2604.04518)
- **Do not trust the classifier's own confidence as a safety signal.** A learned bias
  corrupts confidence first. (2511.07318)
- **Hidden-state distance is blind to this task's main negative.** Changing `1,240` to
  `1,420` barely moves the token distributions but flips the label. Do not use it as the
  detector. (2506.09886)
- **SFT alone may reproduce a weak baseline** where the published gain came from an RL
  stage — RSAT's faithfulness collapsed from 0.97 to 0.03 without its reward. Do not expect
  cited-evidence output from SFT. (2605.00199)
- **Post-hoc attribution does not work.** Bolting citations on after generation dropped
  below 13% format success. If spans are ever wanted, they must be trained in. (2605.00199)
- **An LLM-judged corpus keeps the judge as the weak link** exactly where it is meant to be
  removed. The programmatic tracer is the right call. (2606.16307)
- **World-seed grouping does not dedupe templates.** 60–70% of open-domain QA test answers
  also appear in train; models drop sharply on non-overlapping items. Same-generator
  phrasing overlap is not covered by disjoint seeds.
  Lewis, Stenetorp & Riedel, EACL 2021 — https://aclanthology.org/2021.eacl-main.86/

---

## Numbers and honesty

- **Prevalence.** AUROC and recall-at-fixed-FPR transfer unchanged from a 52/48 synthetic
  set to production. **Precision does not.** At 5% real ungrounded prevalence, recall 0.95
  with FPR 0.049 gives a positive predictive value of roughly **0.50** — half the flags
  would be false. Not in scope for this accuracy-only experiment, but do not let the
  headline number imply otherwise.
  https://www.ncbi.nlm.nih.gov/books/NBK430867/
- **Is the lift real at n = 510?** Detecting a ΔAUC of 0.10 at 80% power needs only 36–142
  per group; ΔAUC of 0.02 needs 909–3,709. The prior Δ was ~0.19. **The existence of the
  lift is not at risk — its cause is.**
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12924612/
- **Test for it properly.** Both models score the same rows, so the comparison is paired:
  McNemar on thresholded decisions, plus a paired bootstrap CI on ΔAUC.
  Dietterich, *Neural Computation* 1998 —
  https://direct.mit.edu/neco/article-abstract/10/7/1895/6224/
- **AUPRC is not automatically better under imbalance** — that claim is contested. Do not
  swap metrics reflexively. https://arxiv.org/pdf/2401.06091

---

## Alternatives considered

1. **Encoder classifier — ModernBERT-large (395M, Apache 2.0, 8192 context) with a sigmoid
   head.**
   *Gains*: covers the p99 4,095-token input in one pass with headroom, no truncation, no
   first-token-logit trick, far cheaper to train, and a direct precedent exists
   (LettuceDetect, RAGTruth example-F1 79.22, MIT).
   *Gives up*: it answers a different question. The experiment asks what fine-tuning does
   to *this* model, and ADR 0001 locks the model.
   *Becomes right if*: Gemma 4 E2B training keeps failing, or the goal shifts from
   "measure the lift" to "ship the fastest gate". Note DeBERTa-v3 is not a candidate — 512
   token limit.
2. **Skip fine-tuning; evaluate Granite Guardian 3.3 8B off the shelf.**
   *Gains*: Apache 2.0, US-origin, already scores 0.831 on RAGTruth and handles
   function-calling hallucination. Zero training.
   *Gives up*: the entire question being asked. 8B is also well above the latency target.
   *Becomes right if*: the fine-tune lift turns out to be an artifact — this is the honest
   fallback.
3. **Confidence routing: cheap model decides, low-confidence rows escalate to the API
   judge.**
   *Gains*: keeps the costly second opinion exactly where false negatives hide.
   *Gives up*: nothing measured here; it is a deployment design, not an experiment.
   *Becomes right if*: this stops being a personal experiment and starts being a gate.

---

## Failure conditions

The recommendation is wrong if any of these turn out true:

- The answer-only baseline scores high. Then the dataset is cheatable and no amount of
  training tells you anything. Fix the data before training again.
- Contrast-consistency is far below row-level accuracy. Then the model reads the edit, not
  the evidence.
- RAGTruth collapses to chance. Then the capability is corpus-specific, and the honest
  report says so.
- No GPU is reachable. Then local MLX is the path, with the four mitigations named in
  [Change 1](#change-1).

## Next actions

1. `prep.py` — copy `data/v1/dev.jsonl`, `data/v1/test.jsonl`, `data/train_v1/sft_train.jsonl`;
   drop the 1,350 `meta.derivation == True` rows; keep `meta.contrast_of` and
   `meta.negative_channel`, they are needed for the controls.
2. `evaluate.py` — score off-the-shelf Gemma 4 E2B on dev. Establish the floor before any
   training happens.
3. Run the **answer-only baseline** on that same off-the-shelf model. If it separates the
   classes, stop and fix the data.
4. `train.py` — LoRA on Unsloth, `unsloth/gemma-4-E2B-it`, `train_on_completions`, 1,000
   rows first.
5. `evaluate.py` — dev sweep, freeze the threshold, then read test **once**. Report AUROC,
   recall at FPR ≤ 5%, contrast-consistency, and the per-channel split.
6. Add RAGTruth as one extra row. Never train on it.
7. `report.md` — the table.

## Open thread

Every paper here checks an answer against **prose passages**. None validates a detector
against nested JSON tool responses where the same number appears in several fields under
different keys. Whether a 2B decoder can reliably resolve "which field licensed this
number" — rather than just "does this number appear anywhere in the blob" — is unanswered
by the read literature. Worth a design checkpoint before trusting a shipped verdict.
