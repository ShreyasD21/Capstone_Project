import numpy as np
import pandas as pd
import pytest

from chess_qnlp.constants import MOTIFS
from chess_qnlp.data.split import assign_splits


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        labels = np.zeros(len(MOTIFS), dtype=int)
        labels[i % len(MOTIFS)] = 1
        if rng.random() < 0.2:
            labels[rng.integers(len(MOTIFS))] = 1
        rows.append({
            "example_id": f"p{i // 2}_{i % 2}",
            "puzzle_id": f"p{i // 2}",                       # two comments per puzzle
            "comment": f"sentence {i // 4}",                 # each sentence shared by two puzzles
            **dict(zip(MOTIFS, labels)),
        })
    return pd.DataFrame(rows)


def test_no_puzzle_or_comment_crosses_splits():
    df = _toy()
    df["split"] = assign_splits(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    assert set(df.split) == {"train", "val", "test"}
    for key in ("puzzle_id", "comment"):
        assert (df.groupby(key).split.nunique() == 1).all(), key


def test_collapsed_grouping_is_rejected():
    df = _toy()
    df["comment"] = [f"sentence {i % 3}" for i in range(len(df))]   # chains every puzzle together
    with pytest.raises(ValueError, match="chaining"):
        assign_splits(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)


def test_every_motif_in_every_split():
    df = _toy()
    df["split"] = assign_splits(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    for name, part in df.groupby("split"):
        assert (part[MOTIFS].sum() > 0).all(), name
