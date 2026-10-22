"""Check that a comment's claims about pieces, colours and squares match the board.

Generated comments pass by construction; this exists for comments from any other
source (LLM-written or human-written), which must be checked before use.
"""

import re

import chess

from chess_qnlp.constants import NAME_TO_PIECE

PIECE_RE = r"(pawn|knight|bishop|rook|queen|king)"


def _exists(boards: list[chess.Board], piece_type: int, color: bool | None = None) -> bool:
    colors = [chess.WHITE, chess.BLACK] if color is None else [color]
    return any(b.pieces(piece_type, c) for b in boards for c in colors)


def verify(comment: str, board: chess.Board, move: chess.Move) -> list[str]:
    """Return the list of problems found; an empty list means the comment passed.

    Claims may refer to the position before or after the key move.
    """
    after = board.copy()
    after.push(move)
    boards = [board, after]
    text = comment.lower()
    problems = []

    for name, square in re.findall(rf"{PIECE_RE} on ([a-h][1-8])", text):
        sq = chess.parse_square(square)
        if not any(b.piece_type_at(sq) == NAME_TO_PIECE[name] for b in boards):
            problems.append(f"no {name} on {square}")

    for color, name in re.findall(rf"\b(white|black) {PIECE_RE}s?\b", text):
        if not _exists(boards, NAME_TO_PIECE[name], color == "white"):
            problems.append(f"no {color} {name}")

    for name in set(re.findall(rf"\b{PIECE_RE}s?\b", text)):
        if not _exists(boards, NAME_TO_PIECE[name]):
            problems.append(f"no {name} on the board")

    return problems
