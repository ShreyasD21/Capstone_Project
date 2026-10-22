"""Extract simple, checkable tactical facts about a key move with python-chess.

These facts are what generated comments are allowed to say, so every comment
built from them is true of the board by construction.
"""

from dataclasses import dataclass

import chess

from chess_qnlp.constants import PIECE_NAMES, PIECE_VALUES

SLIDERS = {chess.BISHOP, chess.ROOK, chess.QUEEN}


@dataclass
class MoveFacts:
    color: str                                  # "white" or "black", the side making the key move
    piece: str                                  # moving piece name
    to_square: str
    captured: str | None = None
    check: bool = False
    fork: tuple[str, str] | None = None         # two valuable targets of the moved piece
    pin: tuple[str, str] | None = None          # (pinned piece, more valuable piece behind it)
    pin_by_mover: bool = False                  # the moved piece is the pinner
    skewer: tuple[str, str] | None = None       # (attacked piece, less valuable piece behind it)
    discovered: tuple[str, str] | None = None   # (revealing piece, newly attacked target)
    sacrifice: bool = False
    hanging: str | None = None                  # undefended piece that was captured


def _value(piece: chess.Piece) -> int:
    return PIECE_VALUES[piece.piece_type]


def _name(piece: chess.Piece) -> str:
    return PIECE_NAMES[piece.piece_type]


def _first_behind(board: chess.Board, origin: int, target: int) -> chess.Piece | None:
    """The first piece on the line from origin through target, beyond target."""
    df = chess.square_file(target) - chess.square_file(origin)
    dr = chess.square_rank(target) - chess.square_rank(origin)
    step_f, step_r = (df > 0) - (df < 0), (dr > 0) - (dr < 0)
    f, r = chess.square_file(target) + step_f, chess.square_rank(target) + step_r
    while 0 <= f < 8 and 0 <= r < 8:
        piece = board.piece_at(chess.square(f, r))
        if piece is not None:
            return piece
        f, r = f + step_f, r + step_r
    return None


def move_facts(board: chess.Board, move: chess.Move) -> MoveFacts:
    mover = board.turn
    piece = board.piece_at(move.from_square)
    captured = board.piece_at(move.to_square)
    if captured is None and board.is_en_passant(move):
        captured = chess.Piece(chess.PAWN, not mover)

    after = board.copy()
    after.push(move)

    facts = MoveFacts(
        color="white" if mover == chess.WHITE else "black",
        piece=_name(piece),
        to_square=chess.square_name(move.to_square),
        captured=_name(captured) if captured else None,
        check=board.gives_check(move),
    )

    # Enemy pieces attacked by the moved piece, most valuable first.
    targets = sorted(
        ((sq, after.piece_at(sq)) for sq in after.attacks(move.to_square)
         if after.piece_at(sq) and after.piece_at(sq).color != mover),
        key=lambda item: _value(item[1]), reverse=True,
    )
    valuable = [p for _, p in targets if _value(p) >= 3]
    if len(valuable) >= 2:
        facts.fork = (_name(valuable[0]), _name(valuable[1]))

    if piece.piece_type in SLIDERS:
        for sq, target in targets:
            behind = _first_behind(after, move.to_square, sq)
            if behind is None or behind.color == mover:
                continue
            if facts.pin is None and _value(behind) > _value(target):
                facts.pin = (_name(target), _name(behind))
                facts.pin_by_mover = True
            if facts.skewer is None and _value(target) > _value(behind) >= 3:
                facts.skewer = (_name(target), _name(behind))

    # Otherwise, an enemy piece pinned to its king before or after the move (the move exploits it).
    if facts.pin is None:
        for b in (board, after):
            pinned = [sq for sq in chess.SQUARES
                      if b.piece_at(sq) and b.piece_at(sq).color != mover
                      and b.piece_type_at(sq) != chess.KING and b.is_pinned(not mover, sq)]
            if pinned:
                facts.pin = (_name(b.piece_at(pinned[0])), "king")
                break

    for sq in after.pieces(chess.BISHOP, mover) | after.pieces(chess.ROOK, mover) | after.pieces(chess.QUEEN, mover):
        if sq == move.to_square:
            continue
        new = after.attacks(sq) & ~board.attacks(sq)
        hit = [after.piece_at(t) for t in new
               if after.piece_at(t) and after.piece_at(t).color != mover and _value(after.piece_at(t)) >= 3]
        if hit:
            facts.discovered = (_name(after.piece_at(sq)), _name(max(hit, key=_value)))
            break

    gained = _value(captured) if captured else 0
    facts.sacrifice = (
        piece.piece_type != chess.KING
        and _value(piece) >= 3
        and _value(piece) > gained
        and after.is_attacked_by(not mover, move.to_square)
    )
    if captured and not board.is_attacked_by(not mover, move.to_square):
        facts.hanging = _name(captured)
    return facts
