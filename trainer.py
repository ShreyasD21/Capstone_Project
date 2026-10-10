import os

import torch
from sklearn.metrics import classification_report

from data_orchestor import run_data_pipeline
from exporter import exporter
from qnlp_model import chessQNLPModel


MOTIF_NAMES = (
    "pin",
    "fork",
    "sacrifice",
    "skewer",
    "deflection",
    "discovered_attack",
    "clearance",
    "hanging_piece",
    "trapped_piece",
    "xray_attack",
    "blunder",
)
NUM_MOTIFS = len(MOTIF_NAMES)


def prepare_split(prepared_data: list):
    """Separate compiled circuits, board vectors, and multi-hot targets."""
    circuits = [item["diagram"] for item in prepared_data]
    boards = torch.stack([item["board_vector"] for item in prepared_data])
    labels = torch.stack([item["label"] for item in prepared_data])
    return circuits, boards, labels


def iter_batches(circuits, boards, labels, batch_size: int):
    for start in range(0, len(circuits), batch_size):
        end = start + batch_size
        yield circuits[start:end], boards[start:end], labels[start:end]


def evaluate_split(model, circuits, boards, labels, criterion, batch_size: int):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_predictions = []
    with torch.no_grad():
        for circuit_batch, board_batch, label_batch in iter_batches(
            circuits, boards, labels, batch_size
        ):
            logits, _ = model(circuit_batch, board_batch)
            loss = criterion(logits, label_batch)
            total_loss += loss.item() * len(circuit_batch)
            total_examples += len(circuit_batch)
            all_predictions.append((torch.sigmoid(logits) >= 0.5).to(torch.int64))

    predictions = torch.cat(all_predictions)
    exact_match = (predictions == labels.to(torch.int64)).all(dim=1).float().mean().item()
    return total_loss / total_examples, exact_match, predictions


def train(
    train_path: str = "data/handoff/train.pkl",
    val_path: str = "data/handoff/val.pkl",
    test_path: str = "data/handoff/test.pkl",
    lr: float = 0.001,
    epochs: int = 10,
    batch_size: int = 8,
):
    train_circuits, train_boards, train_labels = prepare_split(run_data_pipeline(train_path))
    val_circuits, val_boards, val_labels = prepare_split(run_data_pipeline(val_path))
    test_data = run_data_pipeline(test_path)
    test_circuits, test_boards, test_labels = prepare_split(test_data)

    if train_labels.shape[1] != NUM_MOTIFS:
        raise ValueError(
            f"Expected {NUM_MOTIFS} motif targets, got {train_labels.shape[1]}."
        )

    # PennyLaneModel keeps a circuit map internally.  It must know every
    # diagram that may be evaluated, while gradients and labels below still
    # use the training split only.
    all_circuits = train_circuits + val_circuits + test_circuits
    model = chessQNLPModel(
        all_circuits,
        num_classes=NUM_MOTIFS,
        board_dim=train_boards.shape[-1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    positives = train_labels.sum(dim=0)
    negatives = len(train_labels) - positives
    pos_weight = negatives / positives.clamp_min(1)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_examples = 0
        for circuit_batch, board_batch, label_batch in iter_batches(
            train_circuits, train_boards, train_labels, batch_size
        ):
            optimizer.zero_grad()
            train_logits, _ = model(circuit_batch, board_batch)
            train_loss = criterion(train_logits, label_batch)
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item() * len(circuit_batch)
            total_train_examples += len(circuit_batch)

        val_loss, val_exact_match, _ = evaluate_split(
            model, val_circuits, val_boards, val_labels, criterion, batch_size
        )
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train loss: {total_train_loss / total_train_examples:.4f} | "
            f"Validation loss: {val_loss:.4f} | "
            f"Validation exact match: {val_exact_match:.1%}"
        )

    test_loss, test_exact_match, test_predictions = evaluate_split(
        model, test_circuits, test_boards, test_labels, criterion, batch_size
    )
    print("\n--- Final Test Set Performance ---")
    print(f"Test loss: {test_loss:.4f} | Exact-match accuracy: {test_exact_match:.1%}")
    print(
        "\nPer-motif classification report:\n",
        classification_report(
            test_labels.numpy(),
            test_predictions.numpy(),
            target_names=MOTIF_NAMES,
            digits=4,
            zero_division=0,
        ),
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/handoff", exist_ok=True)
    torch.save(model.state_dict(), "models/chess_qnlp_multilabel_model.pt")
    torch.save(exporter(model, test_data), "data/handoff/test_combined_features.pt")
    print("\nModel and test features exported.")
    return model


if __name__ == "__main__":
    train()
