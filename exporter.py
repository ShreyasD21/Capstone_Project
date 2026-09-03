import torch

def exporter(trained_model , prepared_data):
    trained_model.eval() 
    circuits = [item['diagram'] for item in prepared_data]
    board_vectors = torch.stack(item['board_vector'] for item in prepared_data)

    with torch.no_grad():
        quantum_features = trained_model.model(circuits)
        combined_features = torch.cat([quantum_features , board_vectors] , dim=-1)
        
    return combined_features
