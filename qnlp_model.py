import torch
import torch.nn as nn
from lambeq import PennyLaneModel

class chessQNLPModel(nn.Module):
    def __init__(self , compiled_circuits , num_classes : int = 11 , board_dim : int = 768 , q_proj_dim : int = 64 , b_proj_dim : int = 128):
        super().__init__()
        self.model = PennyLaneModel.from_diagrams(compiled_circuits , probabilities=True , normalize=True)
        with torch.no_grad():
            self.model.initialise_weights()
        
        self.num_classes = num_classes
        if not compiled_circuits:
            raise ValueError("At least one compiled circuit is required.")
        # All sentence diagrams have the same sentence output type.  Probing
        # one circuit establishes the probability-vector width without
        # executing every circuit before training starts.
        quantum_dim = self.model([compiled_circuits[0]]).shape[-1]
        self.quantum_projection = nn.Sequential(
            nn.Linear(quantum_dim , q_proj_dim),
            nn.LayerNorm(q_proj_dim),
            nn.GELU()
        )
        self.board_projection = nn.Sequential(
            nn.Linear(board_dim ,b_proj_dim),
            nn.LayerNorm(b_proj_dim),
            nn.GELU()
        )
        fused_dim = b_proj_dim+q_proj_dim;
        self.num_classes = num_classes
        

        self.classifier_head = nn.Linear(fused_dim , num_classes)

    def forward(self , circuits , board_vectors : torch.Tensor):
        raw_quantum_features = self.model(circuits)
        proj_quantum = self.quantum_projection(raw_quantum_features)
        proj_board = self.board_projection(board_vectors)
        fused_features = torch.cat([proj_quantum , proj_board] , dim=-1)
        logits = self.classifier_head(fused_features)
        return logits , raw_quantum_features

    def get_word_embeddings(self):
        return self.model.symbols , self.model.weights
