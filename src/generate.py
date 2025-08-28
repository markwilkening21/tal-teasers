"""
Generate model summaries and save a CSV with human vs model outputs.

Usage (from project root):
    python -m src.generate --split test --batch-size 8
    # or on Colab GPU, with mixed precision:
    python -m src.generate --split test --batch-size 16 --fp16
"""

# --- Hard-disable TF/Flax for Transformers ---
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path
import argparse
from typing import List

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from src import config
from src import data as data_mod
from src import preprocess as pp


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_prefix(model_name: str, texts: List[str]) -> List[str]:
    """Add T5 prefix if needed."""
    return [f"summarize: {t}" for t in texts] if "t5" in model_name.lower() else texts


def main():
    ap = argparse.ArgumentParser(description="Generate model summaries into a CSV.")
    ap.add_argument("--model-dir", type=Path, default=config.MODEL_OUTPUT_DIR,
                    help="Directory with the fine-tuned model (and tokenizer files).")
    ap.add_argument("--split", type=str, choices=["train", "test", "all"], default="test",
                    help="Which data split to generate on.")
    ap.add_argument("--out", type=Path, default=config.GENERATIONS_CSV,
                    help="Output CSV path.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Batch size for generation.")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="Override model.generation_config.max_new_tokens")
    ap.add_argument("--fp16", action="store_true", help="Use float16 on GPU.")
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 on GPU (Ampere+).")
    args = ap.parse_args()

    # --- Load & prepare data ---
    df_raw = data_mod.load_dataframe()                  # requires MERGED_CSV
    df_std = data_mod.standardize_columns(df_raw)       # ensures 'input_text' and 'summary'

    if args.split in ("train", "test"):
        ds = data_mod.make_hf_splits(df_raw, test_size=0.2, seed=config.SEED)
        df_in = ds[args.split].to_pandas()
        # keep these columns only
        df_in = df_in[["episode_id", "url", "title", "input_text", "summary"]]
    else:
        df_in = df_std[["episode_id", "url", "title", "input_text", "summary"]]

    # --- Load model & tokenizer ---
    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.model_dir))
    model.config.use_cache = False  # safer for memory while generating

    # Update generation config if user overrides tokens
    gen_cfg = model.generation_config
    if args.max_new_tokens is not None:
        gen_cfg.max_new_tokens = args.max_new_tokens
    else:
        # Fall back to global config default if missing
        gen_cfg.max_new_tokens = gen_cfg.max_new_tokens or config.MAX_SUMMARY_LEN

    gen_cfg.num_beams = getattr(gen_cfg, "num_beams", None) or config.NUM_BEAMS
    gen_cfg.no_repeat_ngram_size = getattr(gen_cfg, "no_repeat_ngram_size", None) or config.NO_REPEAT_NGRAM_SIZE
    gen_cfg.repetition_penalty = getattr(gen_cfg, "repetition_penalty", None) or config.REPETITION_PENALTY
    gen_cfg.length_penalty = getattr(gen_cfg, "length_penalty", None) or config.LENGTH_PENALTY
    gen_cfg.early_stopping = getattr(gen_cfg, "early_stopping", None)
    if gen_cfg.early_stopping is None:
        gen_cfg.early_stopping = config.EARLY_STOPPING

    device = _device()
    model.to(device)

    # Mixed precision (optional)
    if device.type == "cuda":
        if args.fp16:
            model = model.to(torch.float16)
        elif args.bf16:
            model = model.to(torch.bfloat16)

    # --- Generation loop ---
    texts = _maybe_prefix(str(args.model_dir), df_in["input_text"].astype(str).tolist())
    results = []
    bs = max(1, int(args.batch_size))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for start in tqdm(range(0, len(texts), bs), desc=f"Generating ({args.split})"):
        batch_texts = texts[start:start + bs]

        enc = tok(
            batch_texts,
            max_length=config.MAX_INPUT_LEN,
            truncation=True,
            padding=True,                # pad to longest in batch
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=gen_cfg.max_new_tokens,
                num_beams=gen_cfg.num_beams,
                no_repeat_ngram_size=gen_cfg.no_repeat_ngram_size,
                repetition_penalty=gen_cfg.repetition_penalty,
                length_penalty=gen_cfg.length_penalty,
                early_stopping=gen_cfg.early_stopping,
            )

        decoded = tok.batch_decode(outputs, skip_special_tokens=True)
        for i, pred in enumerate(decoded):
            row_idx = start + i
            results.append({
                "episode_id": int(df_in.iloc[row_idx]["episode_id"]),
                "url":        df_in.iloc[row_idx]["url"],
                "title":      df_in.iloc[row_idx]["title"],
                "human_summary": df_in.iloc[row_idx]["summary"],
                "model_summary": pred.strip(),
            })

    out_df = pd.DataFrame(results, columns=["episode_id", "url", "title", "human_summary", "model_summary"])
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
