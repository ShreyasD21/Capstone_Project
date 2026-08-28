import torch
from qnlpmodel import chessQNLPModel
from data_orchestor import run_data_pipeline

def train(prepared_data: list, num_classes: int = 3):
    compiled_circuits = [item['diagram'] for item in prepared_data]
    
    labels = torch.stack([item['label'] for item in prepared_data])

    model = chessQNLPModel(compiled_circuits, num_classes=num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    logits = model(compiled_circuits)
    loss = criterion(logits, labels)  
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")

if __name__ == "__main__":             
    path = "data.pkl"
    prepared_data = run_data_pipeline(path)
    train(prepared_data, 3)
