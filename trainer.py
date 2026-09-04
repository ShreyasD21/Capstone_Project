import os
import torch
from qnlp_model import chessQNLPModel
from data_orchestor import run_data_pipeline
from torch.utils.data import DataLoader
import torch.nn as nn
from exporter import exporter
def train(train_path : str = "data/handoff/train.pkl" , val_path: str = "data/handoff/val.pkl" , lr : float = 0.001 , epochs : int = 10):
    train_data = run_data_pipeline(train_path)
    val_data = run_data_pipeline(val_path)

    train_circuits = [item['diagram'] for item in train_data]
    train_boards = [item['board_vector'] for item in train_data]
    train_labels = [item['lavel'] for item in train_data]

    val_circuits = [item['diagram'] for item in val_data]
    val_boards = [item['board_vector'] for item in val_data]
    val_labels = [item['label'] for item in val_data]
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

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/handoff", exist_ok=True)
    
    torch.save(model.state_dict(), "models/chess_qnlp_model.pt")
    
    train_fused_features = exporter(model, train_data)
    torch.save(train_fused_features, "data/handoff/train_combined_features.pt")


if __name__ == "__main__":             

    train() 
