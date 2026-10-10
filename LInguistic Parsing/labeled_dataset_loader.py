"""Load multi-label chess-commentary data for the QNLP pipeline."""

from pathlib import Path

import pandas as pd


MOTIF_COLUMNS = (
    "pin",
    "fork",
    "sacrifice",
    "skewer",
    "deflection",
    "discovered_attack",
    "clearance",
    "hanging_piece",
    "trapped_piece",
    "xray_attack",
    "blunder",
)


def load_labeled_dataset(dataset_path: str) -> pd.DataFrame:
    """Load a CSV with a post-move FEN, a comment, and binary motif columns.

    The expected schema is the multi-label dataset format: ``fen`` (or
    ``fen_after``), ``comment``, plus every name in ``MOTIF_COLUMNS``. The
    optional ``fen_before`` and ``move`` fields are retained as provenance.
    """
    path = Path(dataset_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("The labelled dataset must be a CSV file.")

    df = pd.read_csv(path)
    if "fen" not in df.columns and "fen_after" in df.columns:
        df["fen"] = df["fen_after"]

    required = {"fen", "comment", *MOTIF_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.copy()
    for motif in MOTIF_COLUMNS:
        values = pd.to_numeric(df[motif], errors="coerce")
        if values.isna().any() or not values.isin((0, 1)).all():
            raise ValueError(f"'{motif}' must contain only binary values 0 or 1.")
        df[motif] = values.astype(int)

    if (df.loc[:, MOTIF_COLUMNS].sum(axis=1) == 0).any():
        raise ValueError("Every row must have at least one tactical motif label.")
    if "source_file" not in df.columns:
        df["source_file"] = path.name
    return df
