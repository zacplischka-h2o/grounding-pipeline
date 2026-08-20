# Grounding Pipeline

A personal experiment: can a fine-tuned small language model replace an LLM-as-judge
grounding gate, and by how much does fine-tuning move accuracy?

## Language

**Evidence**:
Everything the agent's answer is allowed to rely on. The production sources are: tool call
outputs, tool definitions and schemas, system instructions and prompts, conversation
history, customer details, account data, date-time metadata, and UI components where
relevant.
_Avoid_: context, premise, input

**Response**:
The text of the agent's final answer to the customer. UI components are excluded.
_Avoid_: answer, output, completion

**Grounded**:
A Response is **ungrounded** if any factual component is unsupported, contradicted, or
absent from the Evidence — regardless of how plausible the statement seems. One such
component makes the whole Response ungrounded. There is no partial credit.

Production applies three strictness classes, by statement type:

**Factual statement**:
A claim about the world or the customer. Requires **strict** grounding: every entity,
attribute, value, qualifier, threshold, condition and time period must be explicitly
supported by the outcome of an action, and must match the tool output or computation
result exactly.

**Capability claim**:
A statement about what the agent can do. Must be supported by the system instructions or
the available tool schemas.

**Clarifying question**:
A question the agent asks back. Must correspond to a tool parameter, a workflow
requirement, or a necessary task input.

**Verdict**:
The classifier's binary output: `grounded` or `ungrounded`.
_Avoid_: score, prediction, judgement

**Record**:
One unit of data: Evidence + Response + Verdict label.
_Avoid_: example, sample, row

**False negative**:
An ungrounded Response the classifier passes. The **safety** error — a wrong answer
reaches a banking customer. This is the expensive one.

**False positive**:
A grounded Response the classifier flags. The **latency** error — it triggers a rewrite
and the customer waits longer.

**Judge**:
The current production gate: an API call to a large LLM that returns a Verdict.
The thing this experiment tries to replace.
_Avoid_: baseline, guardrail, gate

**Classifier**:
The fine-tuned small model under test. The candidate replacement for the Judge.

**Negative channel**:
How an ungrounded Record was minted: `minimal-edit` (one small targeted change to a
verified-grounded Response) or `organic` (generated naturally, then labelled).

**Contrast pair**:
A grounded Response and its `minimal-edit` negative. The two differ by a few characters, so
scoring both is the strict test of whether the Classifier reads the evidence or the edit.

**Answer-only baseline**:
The Classifier scored with the Evidence blanked out. If it still separates the classes, the
dataset is cheatable and the headline number means nothing.

**Transfer check**:
An outside dataset, never trained on and never tuned against, read once at the end. It
answers whether a capability learned on synthetic data is real. RAGTruth plays this role.

## ADRs

- [0001 — Non-Chinese base model; Gemma 4 E2B locked](docs/adr/0001-non-chinese-base-model.md)
- [0002 — Grounding means tracing to tool responses](docs/adr/0002-grounding-is-tracing-to-tool-responses.md)
- [0003 — Train on Unsloth, not local MLX](docs/adr/0003-train-on-unsloth-not-local-mlx.md)
- [0004 — Four checks before a number is believed](docs/adr/0004-the-result-is-not-believed-until-four-checks-pass.md)
- [0005 — Script-only baseline runs first](docs/adr/0005-script-only-baseline-runs-before-any-training.md)
- [0006 — RAGTruth Data2txt replaces the synthetic corpus](docs/adr/0006-ragtruth-data2txt-replaces-the-synthetic-corpus.md)
- [0007 — This repo measures factual statements only](docs/adr/0007-this-repo-measures-factual-statements-only.md)
