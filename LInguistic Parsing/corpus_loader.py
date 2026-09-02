from pathlib import Path
import pandas as pd
import chess.pgn
 
def load_corpus(pgn_dir: str) -> pd.DataFrame:
    """Walk a directory of .pgn files, extract every (FEN, comment) pair."""
    rows = []
    for pgn_path in Path(pgn_dir).glob("*.pgn"):
        with open(pgn_path, encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                for node in game.mainline():
                    move = node.move
                    comment = node.comment.strip()
                    board.push(move)
                    if comment: 
                        rows.append({
                            "source_file": pgn_path.name,
                            "move_uci": str(move),
                            "fen": board.fen(),
                            "raw_comment": comment,
                        })
    return pd.DataFrame(rows)
