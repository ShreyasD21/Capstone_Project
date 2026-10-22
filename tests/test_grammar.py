import pytest
from lambeq import AtomicType

from chess_qnlp.data.comments import NAMED, PIN_PASSIVE, UNNAMED
from chess_qnlp.data.masking import mask
from chess_qnlp.text.grammar import GrammarParseError, parse, tokenise

SLOTS = {"p": "knight", "c": "white", "a": "king", "b": "queen"}


def test_tokenise_joins_phrases():
    assert tokenise("The A is pinned to the B.") == ["the", "a", "is pinned to", "the", "b"]


def test_transitive_sentence_has_two_cups():
    diagram = parse("rook pins knight")
    assert diagram.cod == AtomicType.SENTENCE
    assert [b.name for b in diagram.boxes if b.name in ("rook", "pins", "knight")] == ["rook", "pins", "knight"]


@pytest.mark.parametrize("suffix", ["", " with check"])
def test_every_template_and_its_masked_form_parses(suffix):
    templates = [t for group in (NAMED, UNNAMED) for ts in group.values() for t in ts]
    templates += [t for ts in PIN_PASSIVE.values() for t in ts]
    for template in templates:
        sentence = template.format(**SLOTS) + suffix
        for text in (sentence, mask(sentence)):
            assert parse(text).cod == AtomicType.SENTENCE, text


def test_square_modifier_parses():
    assert parse("the knight on f7 forks the king and the queen").cod == AtomicType.SENTENCE


def test_unknown_words_are_reported():
    with pytest.raises(GrammarParseError, match="unknown words: zugzwang"):
        parse("the rook takes zugzwang")


def test_ungrammatical_sentence_fails():
    with pytest.raises(GrammarParseError):
        parse("forks the the king")
