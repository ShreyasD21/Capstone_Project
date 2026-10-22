"""Mini-batch training with early stopping on validation macro-F1."""

import time

import numpy as np
import torch
from torch import nn

from chess_qnlp.evaluate import metrics, tune_thresholds
from chess_qnlp.models.hybrid import HybridModel


def _batches(n: int, size: int, generator: torch.Generator | None = None):
    order = torch.randperm(n, generator=generator) if generator is not None else torch.arange(n)
    for start in range(0, n, size):
        yield order[start:start + size].tolist()


def predict(model: HybridModel, diagrams: list, board: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for idx in _batches(len(board), batch_size):
            logits = model([diagrams[i] for i in idx] if diagrams else [],
                           torch.as_tensor(board[idx]))
            out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out)


def fit(model: HybridModel, data: dict, train_cfg: dict, seed: int, log=print) -> dict:
    """Train in place; return probabilities for every evaluation set plus a training log."""
    y_train = data["y"]["train"]
    pos = y_train.sum(axis=0)
    pos_weight = torch.as_tensor((len(y_train) - pos) / np.maximum(pos, 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    text_params = list(model.text_model.parameters()) if model.text_model is not None else []
    text_ids = {id(p) for p in text_params}
    groups = [{"params": [p for p in model.parameters() if id(p) not in text_ids], "lr": train_cfg["lr_classical"]}]
    if text_params:
        groups.append({"params": text_params, "lr": train_cfg["lr_quantum"]})
    opt = torch.optim.Adam(groups)

    gen = torch.Generator().manual_seed(seed)
    diagrams = data.get("diagrams", {})
    best_f1, best_state, stale, history = -1.0, None, 0, []
    for epoch in range(train_cfg["max_epochs"]):
        start = time.time()
        model.train()
        losses = []
        for idx in _batches(len(y_train), train_cfg["batch_size"], gen):
            batch_diagrams = [diagrams["train"][i] for i in idx] if diagrams else []
            logits = model(batch_diagrams, torch.as_tensor(data["board"]["train"][idx]))
            loss = loss_fn(logits, torch.as_tensor(y_train[idx], dtype=torch.float32))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_prob = predict(model, diagrams.get("val"), data["board"]["val"], train_cfg["batch_size"])
        val_f1 = metrics(data["y"]["val"], val_prob, tune_thresholds(data["y"]["val"], val_prob))["macro_f1"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_macro_f1": val_f1,
                        "seconds": round(time.time() - start, 1)})
        log(f"  epoch {epoch:3d}  loss {np.mean(losses):.4f}  val macro-F1 {val_f1:.3f}  ({history[-1]['seconds']}s)")
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= train_cfg["patience"]:
                log(f"  early stop after epoch {epoch}")
                break

    model.load_state_dict(best_state)
    probs = {name: predict(model, diagrams.get(name), data["board"][name], train_cfg["batch_size"])
             for name in data["eval_sets"]}
    return {"probs": probs, "history": history}
