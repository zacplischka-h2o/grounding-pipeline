# A headline accuracy number is not believed until four checks pass

A prior prototype reported AUROC 0.988 and recall 0.955 at 4.9% FPR — measured on a split
drawn from the same generator family as its training corpus. That is the exact shape of a
model that learned the minting process rather than groundedness, and the literature says it
is the default outcome, not an edge case: SNLI labels are readable from the hypothesis
alone at ~67% (Gururangan et al., NAACL 2018), and style differences between minted and
original text are documented as exploitable shortcuts (Xie et al., KDD 2024).

So this repo does not report a lift on its own. Four checks ride with every number:

1. **Answer-only baseline** — score the test set with the evidence blanked. High separation
   means the dataset is cheatable and training tells us nothing. This is a one-way test: a
   low score is inconclusive (Feng et al., ACL 2019), a high score is a stop signal.
2. **Contrast-consistency** — minimal-edit pairs are scored jointly via `meta.contrast_of`,
   reporting the share of pairs where *both* members are correct. This is the honest number
   and it will sit below row-level accuracy (Gardner et al., EMNLP 2020).
3. **Per-channel split** — `minimal-edit` and `organic` negatives get separate lines. One
   aggregate figure can be carried entirely by the easy half.
4. **RAGTruth as the transfer check** — ~17,790 responses from six LLMs, human-annotated at
   span level, MIT licensed; its data-to-text split feeds structured JSON as evidence, which
   is the nearest public analogue to grounding an answer in tool responses. Read **once**,
   at the end, as one extra row. **Never trained on, never tuned against.**

ADR 0002 dropped ConvFinQA correctly — it tests multi-step arithmetic this agent never
does — but that left no external reference at all, and a synthetic-only result is not
believable alone. RAGTruth tests the right thing: is this claim in the evidence.

RAGTruth is restaurant and news data, not banking, so a lower absolute score there is
expected and is not a failure. The signal is directional: if the fine-tune improves on
RAGTruth too, the capability is real; if it collapses to chance, the synthetic number was
about the generator.

> **Status (2026-08-20): the stop condition fired.** A response-text-only classifier
> reaches test AUROC 0.989 with all digits removed. See
> [the style-separability note](../research/2026-08-20-the-corpus-is-style-separable.md).
