"""LoRA fine-tune Gemma 4 E2B as a groundedness Classifier, and score with it.

Needs a CUDA GPU. See docs/adr/0003-train-on-unsloth-not-local-mlx.md for why this is
not local MLX. Colab T4 is enough (Unsloth reports E2B LoRA at 8-10 GB).

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

# One id for training and for both eval rows. Prequantized 4-bit, never merged.
MODEL_ID = "unsloth/gemma-4-E2B-it"
LOAD_IN_4BIT = True
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


def _strip_bos(tok, text):
    """apply_chat_template emits <bos>; the tokenizer adds another. Unsloth's Gemma 4
    guide says to remove it."""
    return text.removeprefix(tok.bos_token) if tok.bos_token else text


def _prompt_text(tok, prompt):
    return _strip_bos(tok, tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    ))


def first_token_scores(prompts, adapter=None, batch_size=8):
    """P(ungrounded) at the first generated position, for each prompt.

    A real two-way distribution: no sampling, no generation. Returns the normalized
    share of probability mass on the 'ungrounded' first token versus 'grounded'.
    """
    import numpy as np
    import torch

    model, tok = _load(adapter)
    model.eval()

    def variants(word):
        ids = set()
        for form in (word, word.capitalize(), " " + word, " " + word.capitalize()):
            t = tok.encode(form, add_special_tokens=False)
            if t:
                ids.add(t[0])
        return sorted(ids)

    for w in ("grounded", "ungrounded"):
        assert len(tok.encode(w, add_special_tokens=False)) == 1, (
            f"{w!r} is not a single token under this tokenizer. If 'ungrounded' splits "
            f"as 'un' + 'grounded', its first-token mass absorbs every word starting "
            f"'un' — 'Unfortunately', 'Unless', 'Under' — and the readout is noise."
        )
    g_ids, u_ids = variants("grounded"), variants("ungrounded")
    assert not (set(g_ids) & set(u_ids)), (
        "'grounded' and 'ungrounded' share a first token under this tokenizer"
    )

    out = []
    texts = [_prompt_text(tok, p) for p in prompts]
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i : i + batch_size], return_tensors="pt", padding=True,
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
    return Dataset.from_dict({"text": texts})


def main():
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only

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
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
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
        kept = tok.decode([t for t, l in zip(row["input_ids"], row["labels"]) if l != -100])
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
