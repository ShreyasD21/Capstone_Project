import torch
import torch.nn as nn
from lambeq import PennyLaneModel

class chessQNLPModel(nn.Module):
    def __init__(self , compiled_circuits , num_classes : int = None):
        super().__init__()
        self.model = PennyLaneModel.from_diagrams(compiled_circuits , probabilities=True , normalize=True)
        self.model.initialise_weights()
        
        self.num_classes = num_classes
        if num_classes is not None:
            dummy_out = self.model([compiled_circuits[0]])
            quantum_dim = dummy_out.shape[-1]
            self.classifier_head = nn.Linear(quantum_dim, num_classes)
        else:
            self.classifier_head = None
    def forward(self , circuits):
        quantum_features = self.model(circuits)
        if self.classifier_head is not None:
            return self.classifier_head(quantum_features)

        return quantum_features
        
    def get_word_embeddings(self):
        return self.model.symbols , self.model.weights
