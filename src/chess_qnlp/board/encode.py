"""Encode a position and its key move as a 905-dimensional 0/1 vector.

Layout:
    [0, 768)    one-hot pieces: index = 12 * square + plane,
                plane = (piece_type - 1) + (0 if white else 6), squares a1 = 0 ... h8 = 63
    768         side to move (1 = white)
    [769, 833)  from-square one-hot
    [833, 897)  to-square one-hot
    [897, 903)  moving piece type one-hot
    903         move is a capture
    904         move gives check
"""

import chess
import numpy as np

from chess_qnlp.constants import BOARD_DIM


def piece_index(square: int, piece: chess.Piece) -> int:
    plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
    return 12 * square + plane


def encode(board: chess.Board, move: chess.Move) -> np.ndarray:
    v = np.zeros(BOARD_DIM, dtype=np.float32)
    for square, piece in board.piece_map().items():
        v[piece_index(square, piece)] = 1
    v[768] = float(board.turn == chess.WHITE)
    v[769 + move.from_square] = 1
    v[833 + move.to_square] = 1
    v[897 + board.piece_type_at(move.from_square) - 1] = 1
    v[903] = float(board.is_capture(move))
    v[904] = float(board.gives_check(move))
    return v


def encode_rows(fens, moves) -> np.ndarray:
    return np.stack([encode(chess.Board(f), chess.Move.from_uci(m)) for f, m in zip(fens, moves)])
