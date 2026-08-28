from data_loader import ChessQNLPDataset
from data_rewriter import DiagramRewriter

def run_data_pipeline(path : str):
    dataset = ChessQNLPDataset(path)
    rewriter = DiagramRewriter()
    output = []
    for diagram , board_tensor , label_tensor in dataset:
        clean_diagram = rewriter.rewrite_diagram(diagram)
        output.append({'diagram': clean_diagram , 'board_vector':board_tensor ,'label_vector': label_tensor})
    
    print(f"--Successfully proccessed {len(output)} diagrams--")
    return output

if __name__ == "__main__":
    Path = "data.pkl"
    prepared_data = run_data_pipeline(Path)
