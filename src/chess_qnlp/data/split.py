"""Grouped, multi-label-stratified train/val/test split.

Rows sharing a puzzle or an identical comment form one group, and whole groups
are assigned to a split, so no board and no sentence appears in two splits.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from chess_qnlp.constants import MOTIFS


def _groups(df: pd.DataFrame) -> np.ndarray:
    parent = list(range(len(df)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key in ("puzzle_id", "comment"):
        first: dict[str, int] = {}
        for i, value in enumerate(df[key]):
            if value in first:
                parent[find(i)] = find(first[value])
            else:
                first[value] = i
    return np.array([find(i) for i in range(len(df))])


def _stratified(group_ids: np.ndarray, group_labels: np.ndarray, size: float,
                seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=size, random_state=seed)
    keep, hold = next(splitter.split(group_ids.reshape(-1, 1), group_labels))
    return group_ids[keep], group_ids[hold]


def assign_splits(df: pd.DataFrame, fractions: dict, seed: int) -> pd.Series:
    groups = _groups(df)
    labels = df[MOTIFS].to_numpy()
    group_ids = np.unique(groups)
    largest = np.bincount(np.searchsorted(group_ids, groups)).max()
    if len(group_ids) < 20 or largest > 0.1 * len(df):
        raise ValueError(
            f"Grouping by puzzle and comment left {len(group_ids)} groups (largest has {largest} rows). "
            "Shared sentences are chaining puzzles together; generate more varied comments "
            "or fewer comments per puzzle."
        )
    group_labels = np.array([labels[groups == g].max(axis=0) for g in group_ids])

    held = fractions["val"] + fractions["test"]
    train_g, rest_g = _stratified(group_ids, group_labels, held, seed)
    rest_labels = group_labels[np.isin(group_ids, rest_g)]
    val_g, test_g = _stratified(np.sort(rest_g), rest_labels, fractions["test"] / held, seed)

    split = np.empty(len(df), dtype=object)
    split[np.isin(groups, train_g)] = "train"
    split[np.isin(groups, val_g)] = "val"
    split[np.isin(groups, test_g)] = "test"
    return pd.Series(split, index=df.index, name="split")


def save_split_ids(df: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        ids = df.loc[df.split == name, "example_id"]
        (directory / f"{name}_ids.txt").write_text("\n".join(ids) + "\n")


def load_split_ids(directory: Path) -> dict[str, set[str]]:
    return {name: set((directory / f"{name}_ids.txt").read_text().split())
            for name in ("train", "val", "test")}
