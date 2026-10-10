import pandas as pd

from labeled_dataset_loader import MOTIF_COLUMNS
 
def run_eda(df: pd.DataFrame) -> None:
    print("Rows:", len(df))
    print("\nWord count distribution:\n", df["word_count"].describe())
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate FENs (position commented more than once):",
          df["fen"].duplicated().sum())
    if "source_file" in df.columns:
        print("\nComments per source file:\n", df["source_file"].value_counts())
    print("\nPositive examples per motif:\n", df.loc[:, MOTIF_COLUMNS].sum().sort_values())
    print("\nMotifs per example:\n", df.loc[:, MOTIF_COLUMNS].sum(axis=1).value_counts().sort_index())
 
