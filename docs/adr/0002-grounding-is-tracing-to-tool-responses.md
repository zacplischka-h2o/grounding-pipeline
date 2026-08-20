# Grounding means tracing to tool responses, so derivation claims are out of scope

The production agent has a calculator tool and must not do arithmetic in its own head.
Every number it states therefore already appears in a tool response. So the grounding
rule is the simple, script-checkable one: **every claim, and every number, must trace to
the Evidence; if it does not, the Response is ungrounded.**

A prior prototype broke that rule. It added "derivation claims" — grounded answers whose
numbers were computed rather than quoted — because the model failed on **ConvFinQA**, an
external financial-exam dataset full of multi-step arithmetic. ConvFinQA is not this
agent's traffic, and chasing it cost real accuracy (the fine-tuned model reached only
0.253 recall on that slice while hitting 0.955 overall).

Consequences: no derivation stratum in the training corpus, ConvFinQA is not used as a
transfer check, and 1,350 derivation rows are dropped when the prior corpus is reused.

## Amendment (2026-08-20): "or computation results"

The production definition requires a factual statement to match "the tool output **or
computation results**" exactly. That phrase is the one place production language touches
this ADR. Two readings:

- **The calculator's output is a tool output.** Then nothing changes: the number is in the
  Evidence, and this ADR stands as written.
- **The agent may compute and state a number that appears nowhere in the Evidence.** Then
  derivation is back in scope and this ADR is wrong.

**Resolved (2026-08-20): "computation results" means the calculator tool's output.** The
first reading holds. Every number the agent states came back from a tool, so it is in the
Evidence, and this ADR stands unchanged. Derivation stays out of scope.
