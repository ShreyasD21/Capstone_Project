from ansatz_converter import CircuitCompiler

def run_ansatz(output : list , ansatz_type : str = "IQP" , layers : int = 1):
    ansatz = CircuitCompiler(ansatz_type=ansatz_type , l=layers)
    
    ansatz_output = []
    for item in output:
        pqc = ansatz.compile_diagram(item['diagram'])

        ansatz_output.append({
            'circuit': pqc,
            'board_vector': item['board_vector'],
            'label': item['label']
        })
        print(f"Phase 2 Complete: Compiled {len(ansatz_output)} quantum circuits using {ansatz_type} Ansatz.")
        return ansatz_output

if __name__ == "__main__":
    pass
