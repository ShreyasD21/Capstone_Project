"""
Turns the Lichess puzzle CSV (columns: FEN, Moves, Themes) into a small,
verified dataset: one short sentence per position, using only the 5 labels
none / pin / fork / sacrifice / blunder.

Run:
    python build_synthetic_dataset.py --puzzles data/raw/lichess_db_puzzle.csv \
        --out data/synthetic_dataset.csv --per-class 1000
"""
import argparse
import random

import chess
import pandas as pd

# Keep identical to LABEL2ID in motif_labeler.py
LABEL2ID = {"none": 0, "pin": 1, "fork": 2, "sacrifice": 3, "blunder": 4}

NAME = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
        chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
         chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
SIDE = {chess.WHITE: "White", chess.BLACK: "Black"}

TEMPLATES = {
    "pin":       ["The {p} is pinned.", "{side} pins the {p}."],
    "fork":      ["The {p} forks the {a} and the {b}.", "{side} forks the {a} and the {b}."],
    "sacrifice": ["{side} sacrifices the {p}.", "The {p} is sacrificed."],
    "blunder":   ["{side} blunders the {p}.", "The {p} is a blunder."],
    "none":      ["{side} moves the {p}.", "{side} develops the {p}."],
}

# Themes that map to our labels. Rows must contain EXACTLY ONE of these.
THEME2LABEL = {"pin": "pin", "fork": "fork", "sacrifice": "sacrifice",
               "hangingPiece": "blunder"}


def board_after(fen, moves, n):
    """Board after applying the first n UCI moves of the puzzle."""
    b = chess.Board(fen)
    for uci in moves[:n]:
        b.push_uci(uci)
    return b


# ---------------- verifiers: return slot dict or None ----------------
def verify_pin(b):
    """b = board after the solver's move. Opponent (b.turn) has a pinned piece."""
    pinned = [sq for sq, p in b.piece_map().items()
              if p.color == b.turn and p.piece_type != chess.KING
              and b.is_pinned(b.turn, sq)]
    if not pinned:
        return None
    sq = max(pinned, key=lambda s: VALUE[b.piece_at(s).piece_type])
    return {"p": NAME[b.piece_at(sq).piece_type]}


def verify_fork(before, move, after):
    to = move.to_square
    mover = after.piece_at(to)
    if mover is None:
        return None
    enemy = not mover.color
    targets = [after.piece_at(s) for s in after.attacks(to)
               if after.piece_at(s) and after.piece_at(s).color == enemy]
    # a real fork: attacks 2+ pieces that are worth >= the forker, or the king
    good = [t for t in targets if t.piece_type == chess.KING
            or VALUE[t.piece_type] >= VALUE[mover.piece_type]]
    if len(good) < 2:
        return None
    good.sort(key=lambda t: -VALUE[t.piece_type])
    return {"p": NAME[mover.piece_type], "a": NAME[good[0].piece_type],
            "b": NAME[good[1].piece_type]}


def verify_sacrifice(before, move, after):
    piece = before.piece_at(move.from_square)
    if piece is None or VALUE[piece.piece_type] < 3:
        return None
    captured = before.piece_at(move.to_square)
    gain = VALUE[captured.piece_type] if captured else 0
    attacked = after.is_attacked_by(not piece.color, move.to_square)
    if attacked and VALUE[piece.piece_type] - gain >= 2:
        return {"p": NAME[piece.piece_type]}
    return None


def verify_blunder(after_first, move2):
    """after_first = board after the opponent's (blunder) move; solver then captures."""
    if not after_first.is_capture(move2) or after_first.is_en_passant(move2):
        return None
    lost = after_first.piece_at(move2.to_square)
    if lost is None or VALUE[lost.piece_type] < 3:
        return None
    return {"p": NAME[lost.piece_type]}


def make_sentence(label, slots, side):
    return random.choice(TEMPLATES[label]).format(side=side, **slots)


# ---------------- one puzzle row -> (fen, comment, label) or None ----------------
def tactic_row(fen, moves, themes):
    hits = [THEME2LABEL[t] for t in themes if t in THEME2LABEL]
    if len(hits) != 1 or len(moves) < 2:
        return None
    label = hits[0]
    try:
        b0 = chess.Board(fen)
        b1 = board_after(fen, moves, 1)          # after opponent's move
        m2 = chess.Move.from_uci(moves[1])
        b2 = board_after(fen, moves, 2)          # after solver's move
    except ValueError:
        return None
    solver_side = SIDE[b1.turn]                  # side that plays moves[1]
    if label == "pin":
        slots = verify_pin(b2)
    elif label == "fork":
        slots = verify_fork(b1, m2, b2)
    elif label == "sacrifice":
        slots = verify_sacrifice(b1, m2, b2)
    else:  # blunder: the side that played moves[0] lost a piece
        slots = verify_blunder(b1, m2)
        solver_side = SIDE[not b1.turn]
    if slots is None:
        return None
    return b2.fen(), make_sentence(label, slots, solver_side), label


def none_row(fen, moves):
    """Quiet filler: a random non-capturing, non-checking move from the real position."""
    try:
        b = chess.Board(fen)
    except ValueError:
        return None
    quiet = [m for m in b.legal_moves
             if not b.is_capture(m) and not b.gives_check(m)]
    if not quiet:
        return None
    m = random.choice(quiet)
    piece = b.piece_at(m.from_square)
    side = SIDE[b.turn]
    b.push(m)
    slots = {"p": NAME[piece.piece_type]}
    return b.fen(), make_sentence("none", slots, side), "none"


def build(puzzle_csv, out_csv, per_class, seed=42, chunksize=200_000):
    random.seed(seed)
    counts = {k: 0 for k in LABEL2ID}
    rows = []
    for chunk in pd.read_csv(puzzle_csv, usecols=["FEN", "Moves", "Themes"],
                             chunksize=chunksize):
        chunk = chunk.sample(frac=1, random_state=seed)
        for fen, moves, themes in zip(chunk["FEN"], chunk["Moves"], chunk["Themes"]):
            moves, themes = moves.split(), set(themes.split())
            res = tactic_row(fen, moves, themes)
            if res and counts[res[2]] < per_class:
                counts[res[2]] += 1
                rows.append(res)
            elif counts["none"] < per_class and not (themes & set(THEME2LABEL)):
                res = none_row(fen, moves)
                if res:
                    counts["none"] += 1
                    rows.append(res)
            if all(c >= per_class for c in counts.values()):
                break
        if all(c >= per_class for c in counts.values()):
            break
    df = pd.DataFrame(rows, columns=["fen", "comment", "motif_label"])
    df["label_id"] = df["motif_label"].map(LABEL2ID)
    df = df.drop_duplicates(subset=["fen", "comment"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print("Class counts:", df["motif_label"].value_counts().to_dict())
    print("Saved", len(df), "rows ->", out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--out", default="data/synthetic_dataset.csv")
    ap.add_argument("--per-class", type=int, default=1000)
    a = ap.parse_args()
    build(a.puzzles, a.out, a.per_class)
