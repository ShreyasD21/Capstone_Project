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


    model = chessQNLPModel(compiled_circuits, num_classes=num_classes , board_dim=train_boards.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
# epoch implementation
    for epoch in range (1 , epochs+1):
        optimizer.zero_grad()
        logits = model(compiled_circuits)
        loss = criterion(logits, labels)  
        loss.backward()
        optimizer.step()
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean().item() * 100
        
        print(f"Epoch {epoch:02d}/{epoch:02d} | Loss: {loss.item():.4f} | Acc: {acc:.1f}%") 
    torch.save(model.state_dict(), "models/chess_qnlp_model.pt")
    print("Model saved to 'chess_qnlp_model.pt and returned exported model weights")
    return exporter( model , prepared_data)




if __name__ == "__main__":             

 
