"""Replace motif words with neutral words that keep the sentence grammatical."""

import re

MASK = [
    (r"\bforks\b", "attacks"),
    (r"\bfork(ed|ing)?\b", "attack"),
    (r"\bpins\b", "holds"),
    (r"\bpinned\b", "held"),
    (r"\bpin(ning)?\b", "hold"),
    (r"\bskewers\b", "attacks"),
    (r"\bskewer(ed|ing)?\b", "attack"),
    (r"\bdiscovers\b", "opens"),
    (r"\bdiscover(ed|ing)?\b", "open"),
    (r"\bsacrifices\b", "gives"),
    (r"\bsacrificed\b", "given"),
    (r"\bsacrific(e|ing)\b", "give"),
    (r"\bhanging\b", "loose"),
]

KEYWORDS = {
    "fork": r"\bfork",
    "pin": r"\bpin(s|ned|ning)?\b",
    "skewer": r"\bskewer",
    "discoveredAttack": r"\bdiscover",
    "sacrifice": r"\bsacrific",
    "hangingPiece": r"\bhanging\b",
}


def mask(text: str) -> str:
    for pattern, replacement in MASK:
        text = re.sub(pattern, replacement, text)
    return text
