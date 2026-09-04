import torch

def exporter(trained_model , prepared_data):
    trained_model.eval() 
    circuits = [item['diagram'] for item in prepared_data]
    board_vectors = torch.stack(item['board_vector'] for item in prepared_data)

    with torch.no_grad():
        raw_quantum_features = trained_model.model(circuits)
        proj_quantum = trained_model.quantum_projection(raw_quantum_features)
        proj_board = trained_model.board_projection(board_vectors)
        combined_features = torch.cat([proj_quantum , proj_board] , dim=-1)
        
    return combined_features
