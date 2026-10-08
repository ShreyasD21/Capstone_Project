"""Regression tests for weak, rule-based chess motif labels."""

from pathlib import Path
import sys

import pandas as pd


PARSING_DIR = Path(__file__).parent / "LInguistic Parsing"
sys.path.insert(0, str(PARSING_DIR))

from motif_labeler import LABEL2ID, assign_motif_label, label_motifs  # noqa: E402


def test_substrings_do_not_create_false_positive_motifs():
    assert assign_motif_label("Developing the bishop improves the position.") == "none"
    assert assign_motif_label("The position is sacrosanct.") == "none"


def test_supported_inflections_and_annotation_symbol_are_preserved():
    cases = {
        "The knight is pinned to the king.": "pin",
        "White forks the king and rook.": "fork",
        "The pawn is forking both rooks.": "fork",
        "He sacs a bishop.": "sacrifice",
        "A forced sacrifice wins.": "sacrifice",
        "This move is ??.": "blunder",
        "Black loses material.": "blunder",
    }
    for comment, expected_label in cases.items():
        assert assign_motif_label(comment) == expected_label


def test_dataframe_output_uses_stable_label_ids():
    frame = pd.DataFrame({"comment": ["A pin", "Developing normally"]})
    labelled = label_motifs(frame)

    assert labelled["motif_label"].tolist() == ["pin", "none"]
    assert labelled["label_id"].tolist() == [LABEL2ID["pin"], LABEL2ID["none"]]
