import pandas as pd


def load_synthetic_csv(csv_path: str) -> pd.DataFrame:
    """
    Reads the CSV made by build_synthetic_dataset.py
    (columns: fen, comment, motif_label, label_id) and reshapes it so the
    rest of the pipeline (clean_and_align, fen_to_vector, ...) works unchanged.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"comment": "raw_comment"})
    df["source_file"] = "synthetic"
    return df
