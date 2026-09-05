import os
import pytest
import torch
import torch.nn as nn
from lambeq import cups_reader, RemoveCupsRewriter, IQPAnsatz, AtomicType
from qnlp_model import chessQNLPModel

@pytest.fixture
def mock_dataset():
    sentences = ["Rook pins queen", "White blunders knight", "Bishop forks king"]
    type_map = {AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}
    ansatz = IQPAnsatz(type_map, n_layers=1, n_single_qubit_params=3)
    rewriter = RemoveCupsRewriter()
    
    circuits = [ansatz(rewriter(cups_reader.sentence2diagram(s))) for s in sentences]
    boards = torch.randn(3, 768, dtype=torch.float32)
    labels = torch.tensor([1, 4, 2], dtype=torch.long)
    return circuits, boards, labels

def test_forward_and_backward_integrity(mock_dataset):
    circuits, boards, labels = mock_dataset
    model = chessQNLPModel(circuits, board_dim=768, num_classes=5)
    
    logits, quantum_features = model(circuits, boards)
    
    assert logits.shape == (3, 5)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(quantum_features).all()
    
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    
    assert model.board_projection[0].weight.grad is not None
    assert model.quantum_projection[0].weight.grad is not None
    assert model.classifier_head.weight.grad is not None

def test_board_feature_sensitivity(mock_dataset):
    circuits, boards_1, _ = mock_dataset
    boards_2 = boards_1 + 5.0
    
    model = chessQNLPModel(circuits, board_dim=768, num_classes=5)
    model.eval()
    
    with torch.no_grad():
        logits_1, _ = model(circuits, boards_1)
        logits_2, _ = model(circuits, boards_2)
        
    assert not torch.allclose(logits_1, logits_2, atol=1e-3)

def test_tiny_set_overfit(mock_dataset):
    circuits, boards, labels = mock_dataset
    model = chessQNLPModel(circuits, board_dim=768, num_classes=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    
    initial_loss = criterion(model(circuits, boards)[0], labels).item()
    
    for _ in range(25):
        optimizer.zero_grad()
        logits, _ = model(circuits, boards)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
    final_loss = loss.item()
    assert final_loss < initial_loss * 0.2

def test_save_load_reproducibility(mock_dataset, tmp_path):
    circuits, boards, _ = mock_dataset
    model = chessQNLPModel(circuits, board_dim=768, num_classes=5)
    model.eval()
    
    with torch.no_grad():
        orig_logits, _ = model(circuits, boards)
        
    save_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), save_path)
    
    loaded_model = chessQNLPModel(circuits, board_dim=768, num_classes=5)
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.eval()
    
    with torch.no_grad():
        loaded_logits, _ = loaded_model(circuits, boards)
        
    assert torch.equal(orig_logits, loaded_logits)
