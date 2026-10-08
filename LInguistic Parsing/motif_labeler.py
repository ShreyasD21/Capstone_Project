import re

import pandas as pd
 
LABEL2ID = {
    "none": 0,
    "pin": 1,
    "fork": 2,
    "sacrifice": 3,
    "blunder": 4,
}
 
# These are deliberately regexes instead of substring checks.  For example,
# ``"pin" in "developing"`` and ``"sac" in "sacrosanct"`` are true, but
# neither annotation describes a chess motif.  ``\b`` preserves valid word
# forms such as "pinned", "forking", and "sacrificed".
#
# The labeler is single-label.  Rules keep the original priority order, so a
# comment that contains both ``??`` and "sacrifice" is labelled "blunder".
MOTIF_PATTERNS = {
    "blunder": (
        re.compile(r"\?\?"),
        re.compile(r"\bblunder(?:s|ed|ing)?\b"),
        re.compile(r"\bmistake(?:s)?\b"),
        re.compile(r"\blos(?:e|es|t)\s+(?:the\s+)?(?:material|exchange|piece)\b"),
    ),
    "pin": (
        re.compile(r"\bpin(?:s|ned|ning)?\b"),
    ),
    "fork": (
        re.compile(r"\bfork(?:s|ed|ing)?\b"),
    ),
    "sacrifice": (
        re.compile(r"\bsacrifice(?:s|d|ing)?\b"),
        re.compile(r"\bsacs?\b"),
    ),
}


def assign_motif_label(comment: str) -> str:
    """Return the first supported chess motif found in one annotation."""
    text = str(comment).lower()
    for label, patterns in MOTIF_PATTERNS.items():
        if any(pattern.search(text) for pattern in patterns):
            return label
    return "none"

 
def label_motifs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["motif_label"] = df["comment"].apply(assign_motif_label)
    df["label_id"] = df["motif_label"].map(LABEL2ID)
    return df
 
