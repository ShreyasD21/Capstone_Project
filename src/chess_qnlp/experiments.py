"""Run the comparison table: classical baselines, tensor-network and quantum models, ablations.

Usage:
    python -m chess_qnlp.experiments --config configs/default.yaml --only baselines
    python -m chess_qnlp.experiments --config configs/default.yaml --only neural
    python -m chess_qnlp.experiments --config configs/default.yaml --only ablations

Every run writes results/<model>_seed<k>/metrics.json; results/summary.md is rebuilt at the end.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from lambeq import PennyLaneModel, PytorchModel

from chess_qnlp.board.encode import encode_rows
from chess_qnlp.config import ROOT, load_config
from chess_qnlp.constants import BOARD_DIM, MOTIFS
from chess_qnlp.data.split import load_split_ids
from chess_qnlp.evaluate import evaluate_all, save_run, summarise
from chess_qnlp.models.baselines import BASELINES
from chess_qnlp.models.hybrid import HybridModel, n_params
from chess_qnlp.text.circuits import circuit_stats, quantum_ansatz, to_circuit, to_tensor
from chess_qnlp.text.parse import parse_all
from chess_qnlp.train import fit


def load_data(cfg: dict) -> dict:
    """Arrays per evaluation set. 'masked' is the test set with motif words replaced."""
    df = pd.read_csv(cfg["paths"]["processed"] / "examples.csv")
    ids = load_split_ids(cfg["paths"]["splits"])
    parts = {name: df[df.example_id.isin(ids[name])] for name in ("train", "val", "test")}
    parts["masked"] = parts["test"]
    parts["human"] = df[df.split == "human"]
    texts = {name: list(p.comment) for name, p in parts.items()}
    texts["masked"] = list(parts["test"].comment_masked)
    return {
        "frames": parts,
        "eval_sets": ["val", "test", "masked"] + (["human"] if len(parts["human"]) else []),
        "text": texts,
        "y": {name: p[MOTIFS].to_numpy() for name, p in parts.items()},
        "board": {name: encode_rows(p.fen, p.key_move) if len(p) else np.zeros((0, BOARD_DIM), np.float32)
                  for name, p in parts.items()},
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _text_model(kind: str, data: dict, cfg: dict) -> tuple[torch.nn.Module, int]:
    """Build a quantum or tensor-network text model over every sentence it will ever see."""
    sentences = sorted({s for name in ["train", *data["eval_sets"]] for s in data["text"][name]})
    diagrams, errors = parse_all(sentences, cfg["text"], cfg["paths"]["cache"])
    if errors:
        raise RuntimeError(f"{len(errors)} sentences failed to parse; rerun prepare. Example: {next(iter(errors.items()))}")
    if kind == "quantum":
        ansatz = quantum_ansatz(cfg["quantum"])
        converted = {s: to_circuit(diagrams[s], ansatz) for s in sentences}
        # Simulation memory doubles per qubit; never build a model over the budget.
        widest = max(circuit_stats(c)["qubits"] for c in converted.values())
        if widest > cfg["quantum"]["max_sim_qubits"]:
            raise RuntimeError(f"a circuit needs {widest} qubits, over quantum.max_sim_qubits="
                               f"{cfg['quantum']['max_sim_qubits']}; simulating it could exhaust memory")
        model = PennyLaneModel.from_diagrams(list(converted.values()), probabilities=True, normalize=True)
        model.initialise_weights()
        dim = 2 ** cfg["quantum"]["qubits"]["sentence"]
    else:
        converted = {s: to_tensor(diagrams[s], cfg["tensor"]["dims"]) for s in sentences}
        model = PytorchModel.from_diagrams(list(converted.values()))
        model.initialise_weights()
        # lambeq's tensor contraction mixes float64 constants with float32 weights; use float64 throughout.
        model.weights = torch.nn.ParameterList([torch.nn.Parameter(w.detach().double()) for w in model.weights])
        dim = cfg["tensor"]["dims"]["sentence"]
    data["diagrams"] = {name: [converted[s] for s in data["text"][name]] for name in ["train", *data["eval_sets"]]}
    return model, dim


NEURAL = {
    # name: (text model kind, use board)
    "B4_tensor_text": ("tensor", False),
    "Q1_quantum_text": ("quantum", False),
    "B6_tensor_board": ("tensor", True),
    "Q2_quantum_board": ("quantum", True),
}


def run_baselines(cfg: dict, data: dict, results: Path) -> None:
    for name, fn in BASELINES.items():
        for seed in cfg["train"]["seeds"]:
            _seed_all(seed)
            probs = fn(data, seed)
            fixed = np.full(len(MOTIFS), 0.5) if name == "B1_keyword" else None
            out = evaluate_all(data["y"], probs, fixed)
            save_run(results, name, seed, out)
            print(f"{name:20s} seed {seed}  test {out['test']['macro_f1']:.3f}  masked {out['masked']['macro_f1']:.3f}")


def _done(results: Path, name: str, seed: int, eval_sets: list[str]) -> bool:
    """A run is done if its metrics cover every scored set (older runs may lack 'human'; 'val' is never saved)."""
    path = results / f"{name}_seed{seed}" / "metrics.json"
    return path.exists() and all(s in json.loads(path.read_text()) for s in eval_sets if s != "val")


def run_neural(cfg: dict, data: dict, results: Path, models=NEURAL, tag: str = "", resume: bool = False) -> None:
    for name, (kind, use_board) in models.items():
        for seed in cfg["train"]["seeds"]:
            if resume and _done(results, name + tag, seed, data["eval_sets"]):
                print(f"{name + tag} seed {seed}: already done, skipping")
                continue
            _seed_all(seed)
            data = dict(data)
            text_model, dim = _text_model(kind, data, cfg)
            model = HybridModel(text_model, dim, BOARD_DIM if use_board else 0)
            print(f"{name + tag} seed {seed}: {n_params(model)}")
            trained = fit(model, data, cfg["train"], seed)
            out = evaluate_all(data["y"], trained["probs"])
            out.update({"n_params": n_params(model)["total"], "params": n_params(model),
                        "history": trained["history"], "config": {k: cfg[k] for k in ("quantum", "tensor", "train", "text")}})
            save_run(results, name + tag, seed, out)
            print(f"{name + tag:20s} seed {seed}  test {out['test']['macro_f1']:.3f}  masked {out['masked']['macro_f1']:.3f}")


def run_ablations(cfg: dict, data: dict, results: Path, resume: bool = False) -> None:
    for path in sorted((ROOT / "configs" / "ablations").glob("*.yaml")):
        override = yaml.safe_load(path.read_text())
        variant = copy.deepcopy(cfg)
        variant["quantum"].update(override.get("quantum", {}))
        run_neural(variant, data, results, {"Q1_quantum_text": NEURAL["Q1_quantum_text"]}, tag=f"[{path.stem}]",
                   resume=resume)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--only", choices=["baselines", "neural", "ablations", "all"], default="all")
    parser.add_argument("--resume", action="store_true", help="skip neural runs whose metrics.json is complete")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = cfg["paths"]["results"]
    results.mkdir(parents=True, exist_ok=True)
    data = load_data(cfg)
    print({name: len(y) for name, y in data["y"].items()})

    if args.only in ("baselines", "all"):
        run_baselines(cfg, data, results)
    if args.only in ("neural", "all"):
        run_neural(cfg, data, results, resume=args.resume)
    if args.only in ("ablations", "all"):
        run_ablations(cfg, data, results, resume=args.resume)

    table = summarise(results)
    (results / "summary.md").write_text("# Results\n\nMacro-F1, mean ± std over seeds.\n\n" + table)
    print("\n" + table)


if __name__ == "__main__":
    main()
