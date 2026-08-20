# This repo measures factual statements only, not capability claims or clarifying questions

Production defines three strictness classes: **factual statements** (strict — every entity,
attribute, value, qualifier, threshold, condition and time period must match a tool output
or computation result exactly), **capability claims** (must be supported by system
instructions or tool schemas), and **clarifying questions** (must correspond to a tool
parameter, workflow requirement, or necessary task input). See `CONTEXT.md`.

RAGTruth `Data2txt` contains only the first class. Its Records are one-shot overviews
written from a JSON business record: there is no agent, no tool schema, and no dialogue, so
there is nothing for classes 2 and 3 to attach to. Every RAGTruth annotation is a factual
component that is unsupported, contradicted, or absent.

**So this repo measures the fine-tuning lift on factual grounding only.** That is the
largest class and the one the strict rule targets, but a number produced here says nothing
about whether a Classifier can catch an agent promising a transfer its tools cannot make, or
asking for a field no tool takes. Any report must say so.

Two consequences worth stating plainly:

1. **Classes 2 and 3 need a different Evidence shape to test at all** — tool schemas and
   system instructions must be *in* the premise, and RAGTruth has neither. Testing them
   means banking-shaped Records, which means fixing the retired synthetic corpus
   ([ADR 0006](0006-ragtruth-data2txt-replaces-the-synthetic-corpus.md)).
2. **Class 3 is arguably not grounding at all.** A clarifying question asserts nothing, so
   it cannot be unsupported in the sense classes 1 and 2 use. What the rule actually checks
   is whether the question is answerable by the available tools — a relevance and
   capability test wearing a grounding label. Recorded as a disagreement with the
   production taxonomy, not a decision to change anything here.
