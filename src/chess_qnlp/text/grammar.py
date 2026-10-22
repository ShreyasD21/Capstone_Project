"""A small hand-written pregroup grammar for chess comments.

Each word gets one or more pregroup types; a sentence parses if some choice of
types reduces to a single sentence type `s` through planar cancellations
(x·xʳ → 1 and xˡ·x → 1). The reduction found becomes the cups of a lambeq
string diagram.

This replaces lambeq's Bobcat parser, whose model download server no longer
exists. It covers the vocabulary of the generated comments; any other word is
reported as a parse failure rather than guessed.
"""

import itertools
import re
from functools import lru_cache

from lambeq import AtomicType
from lambeq.backend.grammar import Cup, Diagram, Ty, Word

N, S = AtomicType.NOUN, AtomicType.SENTENCE

# Simple types are (base, adjoint order): order -1 = left adjoint, 0 = plain, 1 = right adjoint.
NOUN = (("n", 0),)
DET = (("n", 0), ("n", -1))                                    # n·nˡ   determiner / adjective
TRANSITIVE = (("n", 1), ("s", 0), ("n", -1))                   # nʳ·s·nˡ
INTRANSITIVE = (("n", 1), ("s", 0))                            # nʳ·s
NOUN_PREP = (("n", 1), ("n", 0), ("n", -1))                    # nʳ·n·nˡ  "the knight on f7"
NOUN_POSTMOD = (("n", 1), ("n", 0))                            # nʳ·n     "the rook behind it"
SENT_PREP = (("s", 1), ("s", 0), ("n", -1))                    # sʳ·s·nˡ  "... with check"
SENT_CONJ = (("s", 1), ("s", 0), ("s", -1))                    # sʳ·s·sˡ  "... and ...", "because"
NOUN_CONJ = (("n", 1), ("n", 0), ("n", -1))                    # nʳ·n·nˡ  "the king and the queen"
GERUND = (("n", 0), ("n", -1))                                 # n·nˡ     "moving the knight"

PIECES = ["king", "queen", "rook", "bishop", "knight", "pawn"]
SQUARES = [f + r for f in "abcdefgh" for r in "12345678"]

LEXICON: dict[str, list[tuple]] = {
    **{w: [DET] for w in ["the", "an", "a", "both", "hanging", "undefended", "free", "loose"]},
    **{w: [NOUN] for w in PIECES + SQUARES + ["attack", "check", "it"]},
    "white": [DET, NOUN],
    "black": [DET, NOUN],
    **{w: [TRANSITIVE] for w in [
        "forks", "pins", "skewers", "attacks", "takes", "captures", "wins", "sacrifices",
        "offers", "hits", "discovers", "holds", "opens", "gives", "gives up",
        "is pinned to", "is held to",
    ]},
    **{w: [INTRANSITIVE] for w in [
        "moves", "falls", "moves away", "steps aside", "must move", "cannot move",
        "is sacrificed", "is given", "is behind it",
    ]},
    "moving": [GERUND],
    "behind it": [NOUN_POSTMOD],
    "on": [NOUN_PREP, SENT_PREP],
    "by": [NOUN_PREP],
    "in front of": [NOUN_PREP],
    "with": [SENT_PREP],
    "for": [SENT_PREP],
    "to": [SENT_PREP],
    "and": [NOUN_CONJ, SENT_CONJ],
    "because": [SENT_CONJ],
    "so": [SENT_CONJ],
}

# Articles add qubits but no meaning; they are dropped before building the diagram,
# which is what lambeq's determiner rewrite rule would do.
ARTICLES = {"the", "a", "an"}

# Multi-word expressions are joined into one token, longest first.
PHRASES = sorted((w for w in LEXICON if " " in w), key=len, reverse=True)


class GrammarParseError(ValueError):
    pass


def tokenise(sentence: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", sentence.lower())
    tokens, i = [], 0
    while i < len(words):
        for phrase in PHRASES:
            parts = phrase.split()
            if words[i:i + len(parts)] == parts:
                tokens.append(phrase)
                i += len(parts)
                break
        else:
            tokens.append(words[i])
            i += 1
    return tokens


def _cancels(a: tuple, b: tuple) -> bool:
    return a[0] == b[0] and b[1] == a[1] + 1


def _reduce(types: tuple) -> list[tuple[int, int]] | None:
    """Find planar cups that leave exactly one plain `s`; return the cup index pairs."""

    @lru_cache(maxsize=None)
    def empty(i: int, j: int) -> tuple | None:
        # Cups that cancel types[i:j] completely, or None if impossible.
        if i >= j:
            return ()
        for m in range(i + 1, j):
            if _cancels(types[i], types[m]):
                inner, outer = empty(i + 1, m), empty(m + 1, j)
                if inner is not None and outer is not None:
                    return inner + ((i, m),) + outer
        return None

    for k, t in enumerate(types):
        if t == ("s", 0):
            left, right = empty(0, k), empty(k + 1, len(types))
            if left is not None and right is not None:
                return list(left + right)
    return None


def _ty(simple: tuple) -> Ty:
    base = N if simple[0] == "n" else S
    return {-1: base.l, 0: base, 1: base.r}[simple[1]]


def parse(sentence: str) -> Diagram:
    tokens = [t for t in tokenise(sentence) if t not in ARTICLES]
    unknown = [t for t in tokens if t not in LEXICON]
    if unknown:
        raise GrammarParseError(f"unknown words: {', '.join(unknown)}")
    for choice in itertools.product(*(LEXICON[t] for t in tokens)):
        flat = tuple(simple for word_type in choice for simple in word_type)
        cups = _reduce(flat)
        if cups is not None:
            words = []
            for token, word_type in zip(tokens, choice):
                cod = Ty()
                for simple in word_type:
                    cod = cod @ _ty(simple)
                words.append(Word(token, cod))
            # Inner cups first: create_pregroup_diagram expects adjacent wires at each step.
            morphisms = [(Cup, i, j) for i, j in sorted(cups, key=lambda c: c[1])]
            return Diagram.create_pregroup_diagram(words, morphisms)
    raise GrammarParseError("no grammatical reading reduces to a sentence")
