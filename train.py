"""LoRA fine-tune Gemma 4 E2B as a groundedness Classifier, and score with it.

Needs a CUDA GPU. See docs/adr/0003-train-on-unsloth-not-local-mlx.md for why this is
not local MLX. Tuned for an A100: bf16 weights, real batches. On a T4 or any GPU under
~24GB, set LOAD_IN_4BIT = True and per_device_train_batch_size = 1 (accumulation 16).

    pip install unsloth
    python train.py                 # train, save adapter to models/gemma-ft
    python evaluate.py gemma        # off-the-shelf
    python evaluate.py gemma-ft     # + adapter

evaluate.py imports first_token_scores from here so that both rows are produced by
one loading path. The adapter is never merged: merging shifts the first-token logit
ratio, which is the readout.
"""

import json
import random
from pathlib import Path

# One id for training and for both eval rows. The adapter is never merged.
MODEL_ID = "unsloth/gemma-4-E2B-it"
# bf16, not 4-bit: an A100 has the memory and the hardware bf16 path, so there is no
# reason to eat quantization loss. What matters is that training and BOTH eval rows
# use the same setting — a mismatch would put part of the measured lift in the
# weights rather than the fine-tune. Set True if running on a smaller GPU.
LOAD_IN_4BIT = False
MAX_SEQ = 8192  # 128K-context model; a lower cap only risks silent truncation
ADAPTER = Path("models/gemma-ft")
TRAIN_ROWS = 1000
SEED = 3407

_MODEL = _TOK = _ADAPTER_LOADED = None


def _load(adapter=None):
    """Load once per process. Same weights for every candidate."""
    global _MODEL, _TOK, _ADAPTER_LOADED
    if _MODEL is not None and _ADAPTER_LOADED == adapter:
        return _MODEL, _TOK
    from unsloth import FastModel

    model, tok = FastModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=MAX_SEQ, load_in_4bit=LOAD_IN_4BIT
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))  # attached, not merged
    _MODEL, _TOK, _ADAPTER_LOADED = model, tok, adapter
    return model, tok


def _tok(tok):
    """Resolve the text tokenizer.

    Gemma 4 is multimodal, so FastModel returns a `Gemma4Processor` — a wrapper that
    has `apply_chat_template` and `decode` but NOT `encode`, and does not proxy
    attribute access. The text tokenizer sits at `.tokenizer`. Everything that
    tokenizes or reads a special token goes through here, so training and eval cannot
    diverge on which one they used.
    """
    return getattr(tok, "tokenizer", tok)


def _strip_bos(tok, text):
    """apply_chat_template emits <bos>; the tokenizer adds another. Unsloth's Gemma 4
    guide says to remove it."""
    bos = getattr(_tok(tok), "bos_token", None)
    return text.removeprefix(bos) if bos else text


def _prompt_text(tok, prompt):
    return _strip_bos(tok, tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    ))


def first_token_scores(prompts, adapter=None, batch_size=16):
    """P(ungrounded) at the first generated position, for each prompt.

    A real two-way distribution: no sampling, no generation. Returns the normalized
    share of probability mass on the 'ungrounded' first token versus 'grounded'.
    """
    import numpy as np
    import torch

    model, tok = _load(adapter)
    tk = _tok(tok)
    model.eval()

    def variants(word):
        """First token of the label, and of its capitalized form. Nothing else.

        Neither label is a single token under Gemma 4 — 'grounded' splits as
        ['ground','ed'] and 'ungrounded' as ['ung','rounded'] — but their FIRST tokens
        differ, which is all the readout needs. Space-prefixed forms are deliberately
        excluded: ' ungrounded' begins with ' un', which would also collect
        ' unfortunately', ' unclear', ' unsupported'. The chat template puts the answer
        straight after a newline, so no leading space arises. Both sets are the same
        size, so the ratio is not biased by uneven coverage.
        """
        ids = set()
        for form in (word, word.capitalize()):
            t = tk.encode(form, add_special_tokens=False)
            if t:
                ids.add(t[0])
        return sorted(ids)

    g_ids, u_ids = variants("grounded"), variants("ungrounded")
    assert not (set(g_ids) & set(u_ids)), (
        f"'grounded' {g_ids} and 'ungrounded' {u_ids} share a first token under this "
        f"tokenizer; the readout cannot separate them"
    )
    assert len(g_ids) == len(u_ids), (
        f"asymmetric label coverage: {len(g_ids)} vs {len(u_ids)} token ids. The ratio "
        f"would be biased by which word happens to have more spellings."
    )

    out = []
    texts = [_prompt_text(tok, p) for p in prompts]
    for i in range(0, len(texts), batch_size):
        batch = tk(texts[i : i + batch_size], return_tensors="pt", padding=True,
                   padding_side="left", truncation=True, max_length=MAX_SEQ).to(model.device)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        pg = probs[:, g_ids].sum(-1)
        pu = probs[:, u_ids].sum(-1)
        # A dead two-way distribution divided by a clamp yields a well-behaved number
        # in [0,1] that is pure logit noise, and reads as "weak model" not "broken
        # readout". Stop instead. Checked on every batch: a degenerate minority is
        # invisible in a batch mean.
        assert float((pg + pu).min()) > 1e-4, (
            f"readout dead on a row in batch {i // batch_size}: "
            f"P(grounded)+P(ungrounded) = {float((pg + pu).min()):.2e}. "
            "The prompt was truncated, or the model is not answering with one word."
        )
        out.extend((pu / (pg + pu)).cpu().tolist())
    return np.array(out)


def build_dataset(tok):
    from datasets import Dataset
    from prep import render

    rows = [json.loads(l) for l in open("data/train.jsonl")]
    random.Random(SEED).shuffle(rows)
    rows = rows[:TRAIN_ROWS]
    texts = [
        _strip_bos(tok, tok.apply_chat_template(
            [{"role": "user", "content": render(r)},
             {"role": "assistant", "content": r["label"]}],
            tokenize=False,
        ))
        for r in rows
    ]
    print(f"training rows: {len(texts)}  "
          f"ungrounded {sum(r['label'] == 'ungrounded' for r in rows) / len(rows):.3f}")
    # The prompt is load-bearing and must match eval byte for byte. Show its head so a
    # stray <bos> or a changed template is visible in the log.
    print(f"training text head: {texts[0][:60]!r}")
    return Dataset.from_dict({"text": texts})


def main():
    # unsloth MUST import before trl/transformers/peft — it patches them on import,
    # and a wrong order silently loses the memory optimizations. Not module scope:
    # evaluate.py imports this file to reach first_token_scores, and runs its
    # model-free candidates on machines with no unsloth and no GPU.
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    model, tok = _load()
    model = FastModel.get_peft_model(
        model, r=8, lora_alpha=8, lora_dropout=0, bias="none",
        target_modules="all-linear", random_state=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=build_dataset(tok),
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ,  # TRL defaults to 1024 and truncates from the RIGHT,
            # which would cut off the answer word — the only supervised token.
            # Effective batch stays 16. On 40GB, 4x1 real batches beat 16 accumulation
            # steps by roughly 3x wall clock.
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=2,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=5,
            seed=SEED,
            output_dir="models/checkpoints",
            report_to="none",
        ),
    )

    # Gemma 4's markers, not Gemma 3's <start_of_turn>. A wrong marker masks
    # everything or nothing and the loss cannot tell you: Unsloth documents that a
    # loss of 13-15 is normal for E2B, so it looks fine either way.
    sample = tok.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=False,
    )
    instruction_part, response_part = "<|turn>user\n", "<|turn>model\n"
    assert instruction_part in sample and response_part in sample, (
        f"chat template does not contain the expected Gemma 4 markers.\n"
        f"template renders as:\n{sample}"
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=instruction_part, response_part=response_part
    )

    # Prove the mask is right before spending a GPU hour: what survives must be
    # exactly the answer word plus the turn terminator.
    # Check the LONGEST row, not row 0: a short row passes even when the cap is
    # truncating the answer word off every long one.
    ds = trainer.train_dataset
    longest = max(range(len(ds)), key=lambda i: len(ds[i]["input_ids"]))
    for tag, i in [("row 0", 0), ("longest row", longest)]:
        row = ds[i]
        kept = _tok(tok).decode(
            [t for t, l in zip(row["input_ids"], row["labels"]) if l != -100])
        print(f"supervised tokens of {tag} ({len(row['input_ids'])} tok): {kept!r}")
        assert "grounded" in kept and len(kept) < 40, (
            f"completion-only masking is wrong on {tag}; supervising "
            f"{len(kept)} chars: {kept!r}"
        )

    trainer.train()
    ADAPTER.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER))
    tok.save_pretrained(str(ADAPTER))
    print(f"adapter saved to {ADAPTER}")


if __name__ == "__main__":
    main()
