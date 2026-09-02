import pandas as pd
 
LABEL2ID = {
    "none": 0,
    "pin": 1,
    "fork": 2,
    "sacrifice": 3,
    "blunder": 4,
}
 
# Keywords - just a baseline, not a final labeling strategy.
MOTIF_KEYWORDS = {                                                                 
    "blunder": ["blunder", "??", "loses material", "mistake"],                                   
    "pin": ["pin", "pinned"],
    "fork": ["fork", "forking"],
    "sacrifice": ["sac", "sacrifice", "sacrificing"],
}
 
 
def label_motifs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    def assign_label(comment: str) -> str:
        lower = comment.lower()
        for label, keywords in MOTIF_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return label
        return "none"
 
    df["motif_label"] = df["comment"].apply(assign_label)
    df["label_id"] = df["motif_label"].map(LABEL2ID)
    return df
 
