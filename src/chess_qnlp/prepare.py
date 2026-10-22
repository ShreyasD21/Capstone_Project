"""Build the dataset: puzzles → labels → comments → verification → filtering and parsing → split.

Usage: python -m chess_qnlp.prepare --config configs/default.yaml
"""

import argparse
import json

import chess
import pandas as pd

from chess_qnlp.config import load_config
from chess_qnlp.constants import MOTIFS
from chess_qnlp.data.comments import generate_comments
from chess_qnlp.data.labels import label_vector
from chess_qnlp.data.masking import mask
from chess_qnlp.data.puzzles import load_puzzles
from chess_qnlp.data.split import assign_splits, save_split_ids
from chess_qnlp.data.verify import verify
from chess_qnlp.text.circuits import circuit_stats, quantum_ansatz, to_circuit
from chess_qnlp.text.parse import parse_all


def _verify_rows(df: pd.DataFrame) -> pd.Series:
    return pd.Series([
        "; ".join(verify(r.comment, chess.Board(r.fen), chess.Move.from_uci(r.key_move)))
        for r in df.itertuples()
    ], index=df.index)


def filter_and_parse(df: pd.DataFrame, cfg: dict, counts: dict, dropped: list) -> pd.DataFrame:
    """Shared by generated and human comments: verify, length filter, parse, qubit budget."""
    def drop(mask_, reason):
        nonlocal df
        gone = df[mask_]
        dropped.extend({"example_id": r.example_id, "comment": r.comment, "reason": reason(r)}
                       for r in gone.itertuples())
        df = df[~mask_]

    problems = _verify_rows(df)
    df = df.assign(problems=problems)
    drop(df.problems != "", lambda r: f"verification: {r.problems}")
    counts["after verification"] = len(df)

    n_words = df.comment.str.split().str.len()
    d = cfg["data"]
    drop((n_words < d["min_words"]) | (n_words > d["max_words"]), lambda r: "length")
    counts["after length filter"] = len(df)

    df = df.assign(comment_masked=df.comment.map(mask))
    diagrams, errors = parse_all(list(df.comment) + list(df.comment_masked), cfg["text"], cfg["paths"]["cache"])
    failed = df.comment.isin(errors) | df.comment_masked.isin(errors)
    drop(failed, lambda r: "parse: " + errors.get(r.comment, errors.get(r.comment_masked, "")))
    counts["after parsing"] = len(df)

    ansatz = quantum_ansatz(cfg["quantum"])
    qubits = {s: circuit_stats(to_circuit(diagrams[s], ansatz))["qubits"]
              for s in set(df.comment) | set(df.comment_masked)}
    df = df.assign(qubits=df.comment.map(qubits), qubits_masked=df.comment_masked.map(qubits))
    drop((df.qubits > d["max_qubits"]) | (df.qubits_masked > d["max_qubits"]), lambda r: "qubit budget")
    counts["after qubit budget"] = len(df)
    return df.drop(columns="problems")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    d, paths = cfg["data"], cfg["paths"]
    out = paths["processed"]
    out.mkdir(parents=True, exist_ok=True)

    puzzles = load_puzzles(paths["raw"], d["min_popularity"], d["min_plays"], d["per_motif"],
                           cfg["seed"], d.get("max_rows"))
    labels = pd.DataFrame(puzzles.themes.map(label_vector).tolist(), columns=MOTIFS)
    puzzles = pd.concat([puzzles, labels], axis=1)
    counts = {"puzzles selected": len(puzzles)}

    comments = generate_comments(puzzles, d["comments_per_puzzle"], d["named_fraction"],
                                 d["square_fraction"], cfg["seed"])
    df = comments.merge(puzzles, on="puzzle_id")
    df.insert(0, "example_id", df.puzzle_id + "_" + df.groupby("puzzle_id").cumcount().astype(str))
    counts["comments generated"] = len(df)

    dropped: list[dict] = []
    df = filter_and_parse(df, cfg, counts, dropped)

    df["split"] = assign_splits(df.reset_index(drop=True), d["split"], cfg["seed"]).values
    save_split_ids(df, paths["splits"])
    counts.update({f"{s} examples": int((df.split == s).sum()) for s in ("train", "val", "test")})

    leaked = set(df[df.split == "train"].comment) & set(df[df.split == "test"].comment)
    assert not leaked, f"comments shared between train and test: {list(leaked)[:3]}"

    # Optional human-written comments for test puzzles.
    human_path = paths["human"]
    test_puzzles = puzzles[puzzles.puzzle_id.isin(df[df.split == "test"].puzzle_id)]
    test_puzzles.to_csv(out / "test_puzzles.csv", index=False)
    if human_path.exists():
        human = pd.read_csv(human_path).merge(test_puzzles, on="puzzle_id")
        human.insert(0, "example_id", "human_" + human.index.astype(str))
        human = human.assign(template_motif="human", named=False)
        hcounts: dict = {"human comments": len(human)}
        human = filter_and_parse(human, cfg, hcounts, dropped)
        human["split"] = "human"
        counts.update({f"human: {k}": v for k, v in hcounts.items()})
        df = pd.concat([df, human], ignore_index=True)

    df.to_csv(out / "examples.csv", index=False)
    pd.DataFrame(dropped).to_csv(out / "dropped.csv", index=False)

    named = df[df.split != "human"]
    summary = {
        "counts": counts,
        "named_share": float(named.named.mean()),
        "template_motif": named.template_motif.value_counts().to_dict(),
        "label_counts": {s: named[named.split == s][MOTIFS].sum().astype(int).to_dict()
                         for s in ("train", "val", "test")},
        "qubits": named.qubits.describe().round(2).to_dict(),
        "vocabulary_size": len(set(" ".join(named.comment).split())),
        "drop_reasons": pd.Series([r["reason"].split(":")[0] for r in dropped]).value_counts().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for key, value in counts.items():
        print(f"{key:28s} {value}")
    print(f"\nSaved {out / 'examples.csv'} and split IDs in {paths['splits']}")


if __name__ == "__main__":
    main()
