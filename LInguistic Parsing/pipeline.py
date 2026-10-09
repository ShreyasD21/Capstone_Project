from pathlib import Path

from CSV_loader import load_synthetic_csv
from data_cleaner import clean_and_align
from EDA_report import run_eda
from board_encoder import fen_to_vector
from diagram_parser import parse_to_diagrams, build_type_inventory
from dataset_builder import assemble_triples, split_and_save

# Repo root = the folder above "LInguistic Parsing", so data/ always lands
# next to Sandeep's trainer.py (it reads data/handoff/train.pkl etc.)
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "synthetic_dataset.csv"
OUT_DIR = ROOT / "data" / "handoff"


def run_part1_pipeline(csv_path: str, out_dir: str) -> None:
    # labels are already in the CSV
    df = load_synthetic_csv(csv_path)
    df = clean_and_align(df)
    run_eda(df)

    df["board_vector"] = df["fen"].apply(fen_to_vector)

    diagrams, failed = parse_to_diagrams(df["comment"].tolist())

    inventory = build_type_inventory(diagrams)
    print("\nType inventory (share with Part 2):\n", inventory)

    triples = assemble_triples(df, diagrams)
    print(f"--Kept {len(triples)} / {len(df)} rows that have a diagram--")
    split_and_save(triples, str(out_dir))


if __name__ == "__main__":
    run_part1_pipeline(str(CSV_PATH), str(OUT_DIR))
