from chess_qnlp.constants import MOTIFS
from chess_qnlp.data.labels import label_vector
from chess_qnlp.data.masking import mask


def test_label_vector_order_and_multilabel():
    assert MOTIFS[0] == "fork" and MOTIFS[4] == "sacrifice"
    assert label_vector("fork short middlegame") == [1, 0, 0, 0, 0, 0]
    assert label_vector("sacrifice fork long") == [1, 0, 0, 0, 1, 0]
    assert label_vector("mateIn2 short") == [0, 0, 0, 0, 0, 0]


def test_label_vector_needs_exact_tags():
    # "pin" must not match other theme names that merely contain the letters
    assert label_vector("pinPrevention spinning") == [0] * len(MOTIFS)


def test_mask_removes_every_motif_word():
    cases = {
        "the knight forks the king and the queen": "the knight attacks the king and the queen",
        "the bishop pins the knight to the king": "the bishop holds the knight to the king",
        "the knight is pinned to the queen": "the knight is held to the queen",
        "white sacrifices the rook": "white gives the rook",
        "the queen takes the hanging bishop": "the queen takes the loose bishop",
        "the rook skewers the king and the queen": "the rook attacks the king and the queen",
    }
    for original, expected in cases.items():
        assert mask(original) == expected


def test_mask_leaves_unrelated_words():
    assert mask("an opinion on spinning pieces") == "an opinion on spinning pieces"
