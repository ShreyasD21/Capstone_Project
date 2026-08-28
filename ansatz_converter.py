from lambeq import AtomicType, IQPAnsatz, Sim14Ansatz

class CircuitCompiler:
    def __init__(self , ansatz_type : str = "IQP" , l : int = 1):
        self.type_map = {
            AtomicType.NOUN : 1,
            AtomicType.SENTENCE : 1
        }
        
        if ansatz_type == "IQP":
            self.ansatz = IQPAnsatz(self.type_map , n_layers = l , n_single_qubit_params = 3)
        elif ansatz_type == "Sim14":
            self.ansatz = Sim14Ansatz(self.type_map , n_layers = l)
        else :
            raise ValueError(f"Unsupported ansatz type: {ansatz_type}")

    def compile_diagram(self , diagram):
        return self.ansatz(diagram)
    def compile_batch(self , diagrams):
        return [self.ansatz(d) for d in diagrams]
    
