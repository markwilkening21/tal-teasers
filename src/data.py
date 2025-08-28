"""
Data loading and splitting utilities

Only runs if 'episode_summaries_with_transcripts.csv' exists.
Standardized columns for downstream:
  - 'input_text' (from 'transcript')
  - 'summary'
"""

from pathlib import Path
from typing import Optional
import sys
import pandas as pd
from datasets import Dataset, DatasetDict

from src import config


def load_dataframe(path: Optional[Path] = None) -> pd.DataFrame:
    target = Path(path) if path else Path(config.MERGED_CSV)

    if not target.exists():
        raise FileNotFoundError(
            "Required dataset not found.\n"
            f"Expected: {target}\n"
            "Create it with:  python scripts/combine_csvs.py"
        )

    df = pd.read_csv(target)
    required = {"episode_id", "url", "title", "summary", "transcript"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s) in {target}: {sorted(missing)}\n"
            "Ensure your combine script produced the correct columns."
        )

    df["episode_id"] = pd.to_numeric(df["episode_id"], errors="coerce")
    df = df.dropna(subset=["episode_id", "summary", "transcript"]).copy()
    df["episode_id"] = df["episode_id"].astype(int)

    for col in ("title", "summary", "transcript"):
        df[col] = df[col].astype(str).str.strip()

    df = df.sort_values(["episode_id"]).drop_duplicates(subset=["episode_id"], keep="first")
    return df


def standardize_columns(df: pd.DataFrame,
                        input_col: str = "transcript",
                        target_col: str = "summary") -> pd.DataFrame:
    """
    Map input/target columns to the standardized names used downstream.
    Keeps: episode_id, url, title, input_text, summary
    """
    if input_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Expected columns '{input_col}' and '{target_col}' to exist.")

    keep = ["episode_id", "url", "title", input_col, target_col]
    out = df[keep].rename(columns={input_col: "input_text", target_col: "summary"}).reset_index(drop=True)
    return out


def make_hf_splits(df: pd.DataFrame,
                   test_size: float = 0.2,
                   seed: int = config.SEED) -> DatasetDict:
    """
    Convert a DataFrame to a HF DatasetDict with train/test splits.
    Columns: episode_id, url, title, input_text, summary
    """
    df_std = standardize_columns(df, input_col="transcript", target_col="summary")
    ds_all = Dataset.from_pandas(df_std, preserve_index=False)
    ds = ds_all.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict(train=ds["train"], test=ds["test"])


if __name__ == "__main__":
    # Fail fast if the required file isn't present.
    try:
        frame = load_dataframe()
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        sys.exit(1)

    print(f"Loaded {len(frame)} rows with columns: {list(frame.columns)}")

    ds = make_hf_splits(frame, test_size=0.2, seed=config.SEED)
    print(ds)
    print("Train rows:", len(ds["train"]), "| Test rows:", len(ds["test"]))
