import re

import pandas as pd

MAX_COMMENT_TOKENS = 12


def clean_and_align(df: pd.DataFrame) -> pd.DataFrame:
    """Clean comments and cap circuit width without changing their labels."""

    def strip_artifacts(text: str) -> str:
        text = re.sub(r"\[%.*?\]", "", text)  
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def limit_tokens(text: str) -> str:
        return " ".join(text.split()[:MAX_COMMENT_TOKENS])

    df = df.copy()
    df["comment"] = df["comment"].fillna("").astype(str).apply(strip_artifacts)
    df = df[df["comment"].str.len() > 0]
    df["comment_was_truncated"] = df["comment"].str.split().str.len() > MAX_COMMENT_TOKENS
    df["comment"] = df["comment"].apply(limit_tokens)
    df = df.drop_duplicates(subset=["comment", "fen"])
    df["word_count"] = df["comment"].str.split().str.len()
    return df.reset_index(drop=True)
