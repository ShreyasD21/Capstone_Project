import pickle
import random
from pathlib import Path
 
import pandas as pd
 
 
def assemble_triples(df: pd.DataFrame, diagrams: list) -> list:
    diagram_map = dict(diagrams)                                                          # converts sentences into diagrams
    triples = []
    for _, row in df.iterrows():
        diagram = diagram_map.get(row["comment"])
        if diagram is None:
            continue
        triples.append((diagram, row["board_vector"], row["label_id"]))
    return triples
 
 
def split_and_save(triples: list, out_dir: str,
                    train_frac: float = 0.7, val_frac: float = 0.15,
                    seed: int = 42) -> None:
    random.seed(seed)
    shuffled = triples[:]
    random.shuffle(shuffled)
 
    n = len(shuffled)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
 
    splits = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
 
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split_data in splits.items():
        with open(out / f"{name}.pkl", "wb") as f:
            pickle.dump(split_data, f)
 
    print("Saved splits:", {k: len(v) for k, v in splits.items()})
