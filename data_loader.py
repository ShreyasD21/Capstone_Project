import pickle
import torch
from torch.utils.data import Dataset

class ChessQNLPDataset(Dataset):
    
    def __init__(self , file_path: str):
        with open(file_path , 'rb') as f:
            self.raw_data = pickle.load(f)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self , idx):
        diagram , board_vector , label = self.raw_data[idx]

        board_tensor = torch.tensor(board_vector , dtype=torch.float32)
        label_tensor = torch.tensor(label , dtype=torch.long)

        return diagram, board_tensor , label_tensor
