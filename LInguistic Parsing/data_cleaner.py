import re
import pandas as pd

def clean_and_align(df: pd.DataFrame) -> pd.DataFrame:            #dropping empty and duplicated rows
  def strip_artifacts(text: str) -> str:                          #removing hidden PGN artifacts from comments
        text = re.sub(r"\[%.*?\]", "", text)  
        text = re.sub(r"\s+", " ", text).strip()
        return text
 
    df = df.copy()
    df["comment"] = df["raw_comment"].apply(strip_artifacts)
    df = df[df["comment"].str.len() > 0]
    df = df.drop_duplicates(subset=["comment", "fen"])
    df["word_count"] = df["comment"].str.split().str.len()
    return df.reset_index(drop=True)
 

