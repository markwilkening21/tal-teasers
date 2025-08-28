#!/usr/bin/env python3
"""
Combine two CSVs:
1) episode_summaries_full.csv with columns: episode_id, url, title, summary
2) lines_clean.csv with columns: index, act_name, episode_id, line_text

Output:
A CSV with the first four columns from episode_summaries_full combined
with the full transcripts


"""

import os
import re
from pathlib import Path
import argparse
import pandas as pd

# Try to import config for default paths; fall back to local paths if unavailable.
try:
    from src import config
    DEFAULT_EPISODES = Path(config.DATA_CSV)
    DEFAULT_LINES    = Path(getattr(config, "LINES_CSV", Path(config.DATA_DIR) / "lines_clean.csv"))
    DEFAULT_OUT      = Path(getattr(config, "MERGED_CSV", Path(config.DATA_DIR) / "episode_summaries_with_transcripts.csv"))
except Exception:
    PROJECT_ROOT     = Path(__file__).resolve().parents[1]
    DATA_DIR         = PROJECT_ROOT / "data"
    DEFAULT_EPISODES = DATA_DIR / "episode_summaries_full.csv"
    DEFAULT_LINES    = DATA_DIR / "lines_clean.csv"
    DEFAULT_OUT      = DATA_DIR / "episode_summaries_with_transcripts.csv"


def clean_text(x: str) -> str:
    """Remove newlines/carriage returns and collapse all whitespace to single spaces."""
    s = str(x).replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s)   # collapse multiple spaces/tabs etc.
    return s.strip()


def combine_csvs(episodes_csv: Path, lines_csv: Path, out_csv: Path, sep: str = " ") -> None:
    # Ensure output dir exists
    os.makedirs(out_csv.parent, exist_ok=True)

    # Read inputs
    episodes = pd.read_csv(episodes_csv)
    lines    = pd.read_csv(lines_csv)

    # Ensure consistent dtypes for join key
    episodes["episode_id"] = episodes["episode_id"].astype(int)
    lines["episode_id"]    = lines["episode_id"].astype(int)

    # Normalize line text to avoid embedded newlines
    if "line_text" in lines.columns:
        lines["line_text"] = lines["line_text"].map(clean_text)
    else:
        raise KeyError("Expected 'line_text' column in lines CSV.")

    # Sort lines to preserve order (by index if present; else by act_name)
    if "index" in lines.columns:
        lines = lines.sort_values(["episode_id", "index"])
    elif "act_name" in lines.columns:
        lines = lines.sort_values(["episode_id", "act_name"])

    # Group and concatenate per episode with a single-space separator by default
    grouped = (
        lines.groupby("episode_id")["line_text"]
             .apply(lambda s: sep.join([t for t in s.tolist() if t]))  # filter out empty strings
             .reset_index(name="transcript")
    )

    # Left-join so every episode from summaries is kept
    merged = episodes.merge(grouped, on="episode_id", how="left")
    merged["transcript"] = merged["transcript"].fillna("")

    # Keep order: episode_id, url, title, summary, transcript
    desired_cols = ["episode_id", "url", "title", "summary", "transcript"]
    merged = merged[[c for c in desired_cols if c in merged.columns]]

    merged.to_csv(out_csv, index=False)
    print(f"Wrote {len(merged)} rows to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Combine summaries and per-line transcripts into a single CSV (no newlines in cells).")
    parser.add_argument("--episodes-csv", type=Path, default=DEFAULT_EPISODES,
                        help="Path to episode_summaries_full.csv")
    parser.add_argument("--lines-csv", type=Path, default=DEFAULT_LINES,
                        help="Path to lines_clean.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output CSV path")
    parser.add_argument("--sep", type=str, default=" ",
                        help='Separator to join lines within each episode (default: single space).')
    args = parser.parse_args()
    combine_csvs(args.episodes_csv, args.lines_csv, args.out, args.sep)


if __name__ == "__main__":
    main()
