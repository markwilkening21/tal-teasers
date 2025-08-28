"""
Preprocessing utilities: tokenizer, batch mapping, and data collator.

Assumes upstream data has columns:
  - 'input_text' : model input (e.g., transcript)
  - 'summary'    : gold target text

Typical use:
    from src import data, preprocess, config

    ds = data.make_hf_splits(data.load_dataframe())
    tokenized = preprocess.tokenize_dataset_dict(ds)
    collator = preprocess.make_data_collator()
"""

from typing import Dict, List, Any, Optional
from datasets import DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from src import config

# ---- Tokenizer (cached) -----------------------------------------------------

_tokenizer = None

def get_tokenizer():
    """Load and cache the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT, use_fast=True)
    return _tokenizer


# ---- Helpers ----------------------------------------------------------------

def _needs_t5_prefix() -> bool:
    """Return True if the configured checkpoint looks like a T5 model."""
    return "t5" in str(config.MODEL_CHECKPOINT).lower()

def _maybe_add_prefix(texts: List[str]) -> List[str]:
    """Add 'summarize: ' prefix for T5-style models; else return texts unchanged."""
    if _needs_t5_prefix():
        return [f"summarize: {t}" for t in texts]
    return texts


# ---- Core batch preprocessing -----------------------------------------------

def preprocess_batch(batch: Dict[str, List[str]], tok=None) -> Dict[str, Any]:
    """
    Tokenize a batch dict with keys:
      - 'input_text'
      - 'summary'
    Returns a dict with model inputs and 'labels'.
    """
    tok = tok or get_tokenizer()

    inputs  = batch["input_text"]
    targets = batch["summary"]

    # Add T5 prefix if applicable
    inputs = _maybe_add_prefix(inputs)

    # We rely on dynamic padding via DataCollatorForSeq2Seq, so do not pad here.
    model_inputs = tok(
        inputs,
        max_length=config.MAX_INPUT_LEN,
        truncation=True,
    )

    with tok.as_target_tokenizer():
        labels = tok(
            targets,
            max_length=config.MAX_SUMMARY_LEN,
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_dataset_dict(
    ds: DatasetDict,
    num_proc: Optional[int] = None,
    desc: str = "Tokenizing dataset",
) -> DatasetDict:
    """
    Map `preprocess_batch` over a DatasetDict (train/test).
    Removes original text columns to leave only model-ready tensors.
    """
    tok = get_tokenizer()

    # Grab the text columns from one split (assumed consistent across splits)
    text_cols = [c for c in ds["train"].column_names if c in ("input_text", "summary", "title", "url", "episode_id")]

    tokenized = ds.map(
        lambda batch: preprocess_batch(batch, tok),
        batched=True,
        remove_columns=text_cols,   # keep only tokenized features + labels
        num_proc=num_proc,
        desc=desc,
    )
    return tokenized


# ---- Data collator -----------------------------------------------------------

def make_data_collator(model=None) -> DataCollatorForSeq2Seq:
    """
    Create a collator that:
      - dynamically pads to the longest sequence in a batch
      - pads labels with -100 (ignored by loss)
      - (optionally) knows the model for correct label padding side
    """
    tok = get_tokenizer()
    return DataCollatorForSeq2Seq(
        tokenizer=tok,
        model=model,         
        pad_to_multiple_of=8,
    )


# ---- Quick self-test ---------------------------------------------------------

if __name__ == "__main__":
    # Lightweight smoke test: load data, split, tokenize, print shapes/keys.
    from src import data as data_mod
    ds = data_mod.make_hf_splits(data_mod.load_dataframe(), test_size=0.2, seed=config.SEED)
    print(ds)

    tokenized = tokenize_dataset_dict(ds, desc="Tokenizing (self-test)")
    print(tokenized)
    for split in ("train", "test"):
        batch = tokenized[split][0]
        print(f"{split} sample keys:", list(batch.keys()))
