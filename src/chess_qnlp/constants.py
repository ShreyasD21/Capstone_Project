"""Shared constants. MOTIFS is the single source of truth for label order."""

import chess

# Lichess theme tags used as labels, in label-vector order.
MOTIFS = ["fork", "pin", "skewer", "discoveredAttack", "sacrifice", "hangingPiece"]

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
NAME_TO_PIECE = {name: piece_type for piece_type, name in PIECE_NAMES.items()}

# Rough material values; the king is set high so it always ranks as the most valuable target.
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

COLOR_NAMES = {chess.WHITE: "white", chess.BLACK: "black"}

BOARD_DIM = 905  # see board/encode.py for the layout
