"""Select a balanced subset of Lichess puzzles and resolve the key move.

In the Lichess database, FEN is the position *before* the opponent's move. The first
UCI move is the opponent's; the solver's key move is the second one.
"""

from pathlib import Path

import chess
import numpy as np
import pandas as pd

from chess_qnlp.constants import MOTIFS
from chess_qnlp.data.labels import label_vector

COLUMNS = ["PuzzleId", "FEN", "Moves", "Popularity", "NbPlays", "Themes"]


def read_candidates(raw_path: Path, min_popularity: int, min_plays: int,
                    max_rows: int | None = None, chunksize: int = 500_000) -> pd.DataFrame:
    """Stream the raw CSV and keep well-tested puzzles that carry at least one target motif."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw puzzle file not found: {raw_path}\nRun `bash scripts/download_data.sh` first."
        )
    pattern = r"\b(?:" + "|".join(MOTIFS) + r")\b"
    kept, seen = [], 0
    for chunk in pd.read_csv(raw_path, usecols=COLUMNS, chunksize=chunksize, compression="zstd"):
        chunk = chunk[
            (chunk.Popularity >= min_popularity)
            & (chunk.NbPlays >= min_plays)
            & chunk.Themes.str.contains(pattern, regex=True)
        ]
        kept.append(chunk)
        seen += chunksize
        if max_rows is not None and seen >= max_rows:
            break
    return pd.concat(kept, ignore_index=True)


def sample_balanced(candidates: pd.DataFrame, per_motif: int, seed: int) -> pd.DataFrame:
    """Take up to `per_motif` puzzles per motif, rarest motif first, without repeats."""
    labels = np.array(candidates.Themes.map(label_vector).tolist())
    chosen = np.zeros(len(candidates), dtype=bool)
    rng = np.random.default_rng(seed)
    for i in np.argsort(labels.sum(axis=0)):
        take = per_motif - int(labels[chosen, i].sum())
        pool = np.flatnonzero((labels[:, i] == 1) & ~chosen)
        if take > 0 and len(pool):
            chosen[rng.choice(pool, size=min(take, len(pool)), replace=False)] = True
    return candidates[chosen].reset_index(drop=True)


def resolve_key_move(fen: str, moves: str) -> tuple[str, str]:
    """Return (puzzle position FEN, key move UCI)."""
    uci = moves.split()
    board = chess.Board(fen)
    board.push_uci(uci[0])
    return board.fen(), uci[1]


def load_puzzles(raw_path: Path, min_popularity: int, min_plays: int, per_motif: int,
                 seed: int, max_rows: int | None = None) -> pd.DataFrame:
    candidates = read_candidates(raw_path, min_popularity, min_plays, max_rows)
    puzzles = sample_balanced(candidates, per_motif, seed)
    resolved = [resolve_key_move(f, m) for f, m in zip(puzzles.FEN, puzzles.Moves)]
    return pd.DataFrame({
        "puzzle_id": puzzles.PuzzleId,
        "fen": [r[0] for r in resolved],
        "key_move": [r[1] for r in resolved],
        "themes": puzzles.Themes,
    })
