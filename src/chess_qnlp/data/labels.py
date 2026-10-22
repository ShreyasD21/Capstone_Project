"""Turn Lichess theme tags into multi-label vectors."""

from chess_qnlp.constants import MOTIFS


def label_vector(themes: str) -> list[int]:
    tags = set(themes.split())
    return [int(motif in tags) for motif in MOTIFS]


def has_target_motif(themes: str) -> bool:
    return any(label_vector(themes))
