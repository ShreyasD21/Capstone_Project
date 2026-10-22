"""The hybrid model: a text branch (quantum circuit or classical tensor network) and/or a board branch.

Turning either branch off gives the text-only and board-only ablations through the same code path.
"""

import torch
from torch import nn

from chess_qnlp.constants import MOTIFS


class HybridModel(nn.Module):
    def __init__(self, text_model: nn.Module | None = None, text_dim: int = 0,
                 board_dim: int = 0, n_motifs: int = len(MOTIFS)):
        super().__init__()
        self.text_model = text_model
        fused = 0
        if text_model is not None:
            # Kept small so the expressive work is done by the circuit, not this layer.
            self.text = nn.Sequential(nn.Linear(text_dim, 32), nn.LayerNorm(32), nn.GELU())
            fused += 32
        self.use_board = board_dim > 0
        if self.use_board:
            self.board = nn.Sequential(nn.Linear(board_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2))
            fused += 128
        self.head = nn.Sequential(nn.Linear(fused, 64), nn.GELU(), nn.Linear(64, n_motifs))

    def forward(self, diagrams: list, board: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.text_model is not None:
            out = self.text_model(diagrams).reshape(len(diagrams), -1).float()
            parts.append(self.text(out))
        if self.use_board:
            parts.append(self.board(board))
        return self.head(torch.cat(parts, dim=1))  # logits; the sigmoid lives in the loss


def n_params(model: nn.Module) -> dict:
    text = sum(p.numel() for p in model.text_model.parameters()) if model.text_model is not None else 0
    total = sum(p.numel() for p in model.parameters())
    return {"text_model": text, "classical": total - text, "total": total}
