from ansatz_converter import CircuitCompiler

def run_ansatz(diagram , ansatz_type : str = "IQP" , layers : int = 1):
    ansatz = CircuitCompiler(ansatz_type=ansatz_type , l=layers)
    pqc = ansatz.compile_diagram(diagram)
    return pqc

