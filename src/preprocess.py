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
import os
# Tell Transformers to NOT use TF/Flax at all
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
# (Keep these too; they help on newer versions)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from typing import Dict, List, Any, Optional
from datasets import DatasetDict
from transformers import AutoTokenizer

from src import config

_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT, use_fast=True)
    return _tokenizer

def _with_prefix(xs: List[str]) -> List[str]:
    # T5 expects the "task prefix"
    return [f"summarize: {x}" for x in xs]

def preprocess_batch(batch: Dict[str, List[str]], tok=None) -> Dict[str, Any]:
    """
    Input batch must contain:
      - 'input_text' (source)
      - 'summary'    (targets)
    """
    tok = tok or get_tokenizer()

    inputs_enc = tok(
        _with_prefix(batch["input_text"]),
        padding="max_length",
        truncation=True,
        max_length=config.MAX_INPUT_LEN,
    )

    targets_enc = tok(
        batch["summary"],
        padding="max_length",
        truncation=True,
        max_length=config.MAX_SUMMARY_LEN,
    )

    # Manual label masking: replace pad tokens with -100
    pad_id = tok.pad_token_id
    labels = [
        [(tid if tid != pad_id else -100) for tid in seq]
        for seq in targets_enc["input_ids"]
    ]

    inputs_enc["labels"] = labels
    return inputs_enc

def tokenize_dataset_dict(
    ds: DatasetDict,
    num_proc: Optional[int] = None,
    desc: str = "Tokenizing (max_length, manual label mask)",
) -> DatasetDict:
    tok = get_tokenizer()

    # We remove original text/metadata columns so Trainer sees tensors only
    remove_cols = [c for c in ds["train"].column_names if c in ("episode_id","url","title","input_text","summary")]

    tokenized = ds.map(
        lambda b: preprocess_batch(b, tok),
        batched=True,
        remove_columns=remove_cols,
        num_proc=num_proc,
        desc=desc,
    )
    return tokenized
