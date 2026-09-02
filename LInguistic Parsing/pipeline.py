
from corpus_loader import load_corpus
from data_cleaner import clean_and_align
from eda_report import run_eda
from motif_labeler import label_motifs
from board_encoder import fen_to_vector
from diagram_parser import parse_to_diagrams, build_type_inventory
from dataset_builder import assemble_triples, split_and_save 
 
def run_part1_pipeline(pgn_dir: str, out_dir: str) -> None:
    df = load_corpus(pgn_dir)
    df = clean_and_align(df)
    run_eda(df)
 
    df = label_motifs(df)
    df["board_vector"] = df["fen"].apply(fen_to_vector)
 
    diagrams, failed = parse_to_diagrams(df["comment"].tolist())
    print(f"--Successfully parsed {len(diagrams)} / {len(df)} comments--")
 
    inventory = build_type_inventory(diagrams)
    print("\nType inventory (share with Part 2):\n", inventory)
 
    triples = assemble_triples(df, diagrams)
    split_and_save(triples, out_dir)
 
if __name__ == "__main__":
    run_part1_pipeline(pgn_dir="data/raw_pgn/", out_dir="data/handoff/")
