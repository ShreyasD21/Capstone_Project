import torch
from qnlp_model import chessQNLPModel
from data_orchestor import run_data_pipeline
from torch.utils.data import DataLoader
import torch.nn as nn
from exporter import exporter

def train(prepared_data: list, num_classes: int = 3):
    compiled_circuits = [item['diagram'] for item in prepared_data]
    
    labels = torch.stack([item['label'] for item in prepared_data])

    model = chessQNLPModel(compiled_circuits, num_classes=num_classes)
    epochs = 6
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
    path = "data.pkl"
    prepared_data = run_data_pipeline(path)
    torch.save(train(prepared_data, 4) , "handoff/combined_features.pt")

