import os
import torch
from sklearn.metrics import classification_report, confusion_matrix
from qnlp_model import chessQNLPModel
from data_orchestor import run_data_pipeline
from torch.utils.data import DataLoader
import torch.nn as nn
from exporter import exporter
NUM_ClASSES = 5

def evaluate_split(model, circuits, boards, labels, criterion):
    model.eval()
    with torch.no_grad():
        logits, _ = model(circuits, boards)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean().item() * 100
    return loss.item(), acc, preds

def train(train_path : str = "data/handoff/train.pkl" , val_path: str = "data/handoff/val.pkl" , test_path : str = "data/handoff/test.pkl" , lr : float = 0.001 , epochs : int = 10):
    train_data = run_data_pipeline(train_path)
    val_data = run_data_pipeline(val_path)
    test_data = run_data_pipeline(test_path)

    train_circuits = [item['diagram'] for item in train_data]
    train_boards = [item['board_vector'] for item in train_data]
    train_labels = [item['label'] for item in train_data]

    val_circuits = [item['diagram'] for item in val_data]
    val_boards = [item['board_vector'] for item in val_data]
    val_labels = [item['label'] for item in val_data]

    test_circuits = [item['diagram'] for item in test_data]
    test_boards = torch.stack([item['board_vector'] for item in test_data])
    test_labels = torch.stack([item['label'] for item in test_data])

    num_classes = int(torch.max(torch.cat([train_labels, val_labels])).item() + 1)

    model = chessQNLPModel(train_circuits ,  num_classes=num_classes , board_dim=train_boards.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
# epoch implementation
    for epoch in range (1 , epochs+1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_circuits , train_boards)
        loss = criterion(logits, train_labels)  
        loss.backward()
        optimizer.step()
        train_acc = (torch.argmax(logits, dim=-1) == train_labels).float().mean().item() * 100
        
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(val_circuits, val_boards)
            val_loss = criterion(val_logits, val_labels)
            val_acc = (torch.argmax(val_logits, dim=-1) == val_labels).float().mean().item() * 100
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {loss.item():.4f} (Acc: {train_acc:.1f}%) | Val Loss: {val_loss.item():.4f} (Acc: {val_acc:.1f}%)")        

    test_loss, test_acc, test_preds = evaluate_split(model, test_circuits, test_boards, test_labels, criterion)
    print("\n--- Final Test Set Performance ---")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print("\nClassification Report:\n", classification_report(test_labels.numpy(), test_preds.numpy(), digits=4))
    print("Confusion Matrix:\n", confusion_matrix(test_labels.numpy(), test_preds.numpy()))

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/handoff", exist_ok=True)
    
    torch.save(model.state_dict(), "models/chess_qnlp_model.pt")
    
    torch.save(exporter(model, train_data), "data/handoff/train_combined_features.pt")
    torch.save(exporter(model, val_data), "data/handoff/val_combined_features.pt")
    torch.save(exporter(model, test_data), "data/handoff/test_combined_features.pt")
    
    print("\n✅ Models and split features successfully exported.")


if __name__ == "__main__":             

    train() 
