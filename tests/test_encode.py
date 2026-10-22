import chess

from chess_qnlp.board.encode import encode, piece_index
from chess_qnlp.constants import BOARD_DIM


def test_piece_index_examples():
    assert piece_index(chess.A1, chess.Piece(chess.ROOK, chess.WHITE)) == 3
    assert piece_index(chess.E1, chess.Piece(chess.KING, chess.WHITE)) == 53
    assert piece_index(chess.E8, chess.Piece(chess.KING, chess.BLACK)) == 731


def test_start_position():
    board = chess.Board()
    v = encode(board, chess.Move.from_uci("g1f3"))
    assert v.shape == (BOARD_DIM,)
    assert v[:768].sum() == 32
    assert v[768] == 1                       # white to move
    assert v[769 + chess.G1] == 1
    assert v[833 + chess.F3] == 1
    assert v[897 + chess.KNIGHT - 1] == 1
    assert v[903] == 0 and v[904] == 0


def test_side_to_move_and_check():
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 b - - 0 1")
    v = encode(board, chess.Move.from_uci("e8d7"))
    assert v[768] == 0

    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    v = encode(board, chess.Move.from_uci("a1a8"))
    assert v[904] == 1
