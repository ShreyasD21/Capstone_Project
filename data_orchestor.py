from data_loader import ChessQNLPDataset
from data_rewriter import DiagramRewriter
from ansatz_orchestor import run_ansatz

def run_data_pipeline(path : str):
    dataset = ChessQNLPDataset(path)
    rewriter = DiagramRewriter()
    output = []
    for diagram , board_tensor , label_tensor in dataset:
        clean_diagram = rewriter.rewrite_diagram(diagram)
        compiled_diagram = run_ansatz(clean_diagram , "IQP" , 1)
        output.append({'diagram': compiled_diagram , 'board_vector':board_tensor ,'label': label_tensor})
    print(f"--Successfully proccessed {len(output)} diagrams--")
    return output


