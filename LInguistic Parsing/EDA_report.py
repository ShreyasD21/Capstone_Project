import pandas as pd 
 
def run_eda(df: pd.DataFrame) -> None:
    print("Rows:", len(df))
    print("\nWord count distribution:\n", df["word_count"].describe())
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate FENs (position commented more than once):",
          df["fen"].duplicated().sum())
    print("\nComments per source file:\n", df["source_file"].value_counts())
 
