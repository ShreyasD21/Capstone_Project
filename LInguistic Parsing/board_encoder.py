import chess
import numpy as np 
 
def fen_to_vector(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    vector = np.zeros(768, dtype=np.float32)
    for square in range(64):
        piece = board.piece_at(square)
        if piece is None:
            continue
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        vector[square * 12 + plane] = 1.0
    return vector
 
