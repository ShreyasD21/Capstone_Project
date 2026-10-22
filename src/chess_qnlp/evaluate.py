"""Metrics, threshold tuning, and the results summary table.

Run `python -m chess_qnlp.evaluate --summarise results/` to rebuild results/summary.md.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from chess_qnlp.constants import MOTIFS

GRID = np.linspace(0.05, 0.95, 19)


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-motif threshold maximising F1 on validation data."""
    best = []
    for j in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, j], y_prob[:, j] >= t, zero_division=0) for t in GRID]
        best.append(GRID[int(np.argmax(scores))])
    return np.array(best)


def metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> dict:
    y_pred = (y_prob >= thresholds).astype(int)
    per_motif = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match": float((y_pred == y_true).all(axis=1).mean()),
        "per_motif_f1": {m: float(f) for m, f in zip(MOTIFS, per_motif)},
    }


def evaluate_all(y: dict[str, np.ndarray], probs: dict[str, np.ndarray],
                 thresholds: np.ndarray | None = None) -> dict:
    """Score every available test set; thresholds are tuned on 'val' unless given."""
    if thresholds is None:
        thresholds = tune_thresholds(y["val"], probs["val"])
    out = {"thresholds": thresholds.tolist()}
    for name in probs:
        if name != "val" and len(y[name]):
            out[name] = metrics(y[name], probs[name], thresholds)
    return out


def save_run(results_dir: Path, model: str, seed: int, payload: dict) -> Path:
    run_dir = results_dir / f"{model}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps({"model": model, "seed": seed, **payload}, indent=2))
    return run_dir


def summarise(results_dir: Path) -> str:
    runs = defaultdict(lambda: defaultdict(list))
    params = {}
    for path in sorted(results_dir.glob("*/metrics.json")):
        run = json.loads(path.read_text())
        for test_set in ("test", "masked", "human"):
            if test_set in run:
                runs[run["model"]][test_set].append(run[test_set]["macro_f1"])
        params[run["model"]] = run.get("n_params", "")

    def cell(values: list[float]) -> str:
        return f"{np.mean(values):.3f} ± {np.std(values):.3f} (n={len(values)})" if values else "—"

    lines = ["| Model | Params | Test macro-F1 | Masked macro-F1 | Human macro-F1 |",
             "|---|---|---|---|---|"]
    for model in sorted(runs):
        r = runs[model]
        lines.append(f"| {model} | {params[model]} | {cell(r['test'])} | {cell(r['masked'])} | {cell(r['human'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarise", type=Path, required=True, help="results directory")
    args = parser.parse_args()
    table = summarise(args.summarise)
    (args.summarise / "summary.md").write_text("# Results\n\nMean ± std over seeds.\n\n" + table)
    print(table)


if __name__ == "__main__":
    main()
