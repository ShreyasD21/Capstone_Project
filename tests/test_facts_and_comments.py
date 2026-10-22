import random

import chess

from chess_qnlp.data.comments import generate_comment
from chess_qnlp.data.facts import move_facts
from chess_qnlp.data.verify import verify


def test_knight_fork():
    # Nxf7+ forks the king on h8 and the queen on d8
    board = chess.Board("3q3k/5p2/8/4N3/8/8/8/6K1 w - - 0 1")
    facts = move_facts(board, chess.Move.from_uci("e5f7"))
    assert facts.piece == "knight"
    assert facts.fork == ("king", "queen")
    assert facts.check and facts.captured == "pawn"


def test_bishop_pin():
    # Bb5 pins the knight on c6 to the king on e8
    board = chess.Board("4k3/8/2n5/8/8/8/8/3BK3 w - - 0 1")
    facts = move_facts(board, chess.Move.from_uci("d1b3"))
    assert facts.pin is None
    board = chess.Board("4k3/8/2n5/8/8/8/8/4KB2 w - - 0 1")
    facts = move_facts(board, chess.Move.from_uci("f1b5"))
    assert facts.pin == ("knight", "king")


def test_rook_skewer():
    # Re1+ skewers the king on e5 and the queen on e8
    board = chess.Board("4q3/8/8/4k3/8/8/8/R5K1 w - - 0 1")
    facts = move_facts(board, chess.Move.from_uci("a1e1"))
    assert facts.skewer == ("king", "queen")


def test_hanging_piece():
    board = chess.Board("4k3/8/8/8/2b5/8/8/2R1K3 w - - 0 1")
    facts = move_facts(board, chess.Move.from_uci("c1c4"))
    assert facts.hanging == "bishop"


def test_generated_comments_pass_verification():
    board = chess.Board("3q3k/5p2/8/4N3/8/8/8/6K1 w - - 0 1")
    move = chess.Move.from_uci("e5f7")
    rng = random.Random(0)
    for _ in range(50):
        row = generate_comment(board, move, "fork short", rng, named_fraction=0.5, square_fraction=0.5)
        assert row["template_motif"] == "fork"
        assert verify(row["comment"], board, move) == [], row["comment"]


def test_verify_catches_false_claims():
    board = chess.Board("3q3k/5p2/8/4N3/8/8/8/6K1 w - - 0 1")
    move = chess.Move.from_uci("e5f7")
    assert verify("the knight on f7 forks the king and the queen", board, move) == []
    assert "no bishop on g3" in verify("the bishop on g3 is attacked", board, move)
    assert "no black knight" in verify("the black knight is trapped", board, move)
    assert "no rook on the board" in verify("the knight forks the king and the rook", board, move)
