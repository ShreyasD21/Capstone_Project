from data_loader import ChessQNLPDataset
from data_rewriter import DiagramRewriter
from ansatz_converter import CircuitCompiler


def run_data_pipeline(path: str):
    """Rewrite and compile one saved split into PennyLane-ready circuits."""
    dataset = ChessQNLPDataset(path)
    rewriter = DiagramRewriter()
    compiler = CircuitCompiler(ansatz_type="IQP", l=1)
    output = []
    for diagram, board_tensor, label_tensor in dataset:
        clean_diagram = rewriter.rewrite_diagram(diagram)
        compiled_diagram = compiler.compile_diagram(clean_diagram)
        output.append({
            "diagram": compiled_diagram,
            "board_vector": board_tensor,
            "label": label_tensor,
        })
    print(f"Successfully processed {len(output)} diagrams.")
    return output

