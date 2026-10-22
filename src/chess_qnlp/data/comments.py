"""Generate short English comments from verified move facts.

Each motif has "named" templates (the motif word appears) and "unnamed" ones
(the tactic is described without its keyword), so the masked test is meaningful.
When none of a puzzle's motifs can be described from the facts, a generic
description of the move is used instead.
"""

import random

import chess
import pandas as pd

from chess_qnlp.constants import MOTIFS
from chess_qnlp.data.facts import MoveFacts, move_facts
from chess_qnlp.data.labels import label_vector

NAMED = {
    "fork": ["the {p} forks the {a} and the {b}",
             "the {c} {p} forks the {a} and the {b}"],
    "pin": ["the {p} pins the {a} to the {b}",
            "the {a} is pinned to the {b}"],
    "skewer": ["the {p} skewers the {a} and the {b}",
               "the {c} {p} skewers the {a} and the {b}"],
    "discoveredAttack": ["the {p} discovers an attack on the {b}",
                         "moving the {p} discovers an attack by the {a}"],
    "sacrifice": ["{c} sacrifices the {p}",
                  "the {c} {p} is sacrificed"],
    "hangingPiece": ["the {p} takes the hanging {a}",
                     "the {c} {p} captures the hanging {a}"],
}

UNNAMED = {
    "fork": ["the {p} attacks the {a} and the {b}",
             "the {c} {p} attacks both the {a} and the {b}"],
    "pin": ["the {a} cannot move because the {b} is behind it",
            "the {p} attacks the {a} in front of the {b}"],
    "skewer": ["the {p} attacks the {a} and the {b} behind it falls",
               "the {a} must move and the {b} behind it falls"],
    "discoveredAttack": ["the {p} moves away and the {a} attacks the {b}",
                         "the {p} steps aside so the {a} hits the {b}"],
    "sacrifice": ["{c} gives up the {p}",
                  "{c} offers the {p} for an attack"],
    "hangingPiece": ["the {p} takes the undefended {a}",
                     "the {c} {p} wins the free {a}"],
}

GENERIC_CAPTURE = ["the {p} takes the {a}", "the {c} {p} captures the {a}"]
GENERIC_MOVE = ["the {c} {p} moves", "the {p} moves to {sq}"]

# Pin templates that don't say which piece is pinning, for pins the move exploits.
PIN_PASSIVE = {True: ["the {a} is pinned to the {b}"],
               False: ["the {a} cannot move because the {b} is behind it"]}


def _slots(motif: str, f: MoveFacts) -> dict | None:
    """Template slot values for a motif, or None if the facts can't support it."""
    base = {"p": f.piece, "c": f.color}
    if motif == "fork" and f.fork:
        return base | {"a": f.fork[0], "b": f.fork[1]}
    if motif == "pin" and f.pin:
        return base | {"a": f.pin[0], "b": f.pin[1]}
    if motif == "skewer" and f.skewer:
        return base | {"a": f.skewer[0], "b": f.skewer[1]}
    if motif == "discoveredAttack" and f.discovered:
        return base | {"a": f.discovered[0], "b": f.discovered[1]}
    if motif == "sacrifice" and f.sacrifice:
        return base
    if motif == "hangingPiece" and f.hanging:
        return base | {"a": f.hanging}
    return None


def _with_square(text: str, f: MoveFacts) -> str:
    """Mention the destination square when the sentence starts with the moving piece."""
    if f.to_square in text:
        return text
    for phrase in (f"the {f.piece}", f"the {f.color} {f.piece}"):
        if text.startswith(phrase + " "):
            return text.replace(phrase, f"{phrase} on {f.to_square}", 1)
    return text


def generate_comment(board: chess.Board, move: chess.Move, themes: str,
                     rng: random.Random, named_fraction: float, square_fraction: float) -> dict:
    facts = move_facts(board, move)
    labels = label_vector(themes)
    candidates = [m for m, y in zip(MOTIFS, labels) if y and _slots(m, facts)]
    if candidates:
        motif = rng.choice(candidates)
        named = rng.random() < named_fraction
        options = (NAMED if named else UNNAMED)[motif]
        if motif == "pin" and not facts.pin_by_mover:
            options = PIN_PASSIVE[named]
        text = rng.choice(options).format(**_slots(motif, facts))
    else:
        motif, named = "generic", False
        if facts.captured:
            text = rng.choice(GENERIC_CAPTURE).format(p=facts.piece, c=facts.color, a=facts.captured)
        else:
            text = rng.choice(GENERIC_MOVE).format(p=facts.piece, c=facts.color, sq=facts.to_square)
    if facts.check and rng.random() < 0.5:
        text += " with check"
    if rng.random() < square_fraction:
        text = _with_square(text, facts)
    return {"comment": text, "template_motif": motif, "named": named}


def generate_comments(puzzles: pd.DataFrame, per_puzzle: int, named_fraction: float,
                      square_fraction: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for p in puzzles.itertuples():
        board = chess.Board(p.fen)
        move = chess.Move.from_uci(p.key_move)
        seen = set()
        for _ in range(per_puzzle):
            row = generate_comment(board, move, p.themes, rng, named_fraction, square_fraction)
            if row["comment"] in seen:
                continue
            seen.add(row["comment"])
            rows.append({"puzzle_id": p.puzzle_id, **row})
    return pd.DataFrame(rows)
