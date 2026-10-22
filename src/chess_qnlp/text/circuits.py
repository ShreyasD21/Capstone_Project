"""Turn string diagrams into quantum circuits or classical tensor networks."""

from lambeq import AtomicType, IQPAnsatz, RemoveCupsRewriter, Rewriter, TensorAnsatz
from lambeq.backend import quantum
from lambeq.backend.grammar import Diagram
from lambeq.backend.tensor import Dim

N, S = AtomicType.NOUN, AtomicType.SENTENCE

# Sim14 is left out: with cups removed, lambeq 0.5 keeps negated symbols on flipped boxes
# (KeyError "-1.0 word__type_0" in PennyLaneModel), and with cups kept circuits reach
# ~22 qubits, which needs tens of GB of memory to simulate.
ANSATZE = {"iqp": IQPAnsatz}

_simplify = Rewriter(["determiner"])
_remove_cups = RemoveCupsRewriter()


def simplify(diagram: Diagram) -> Diagram:
    """Turn determiners into plain wires (fewer qubits, same meaning).

    The project grammar already drops articles, so this only changes Bobcat diagrams.
    """
    if any(box.name in ("the", "a", "an") for box in diagram.boxes):
        return _simplify(diagram)
    return diagram


def quantum_ansatz(qcfg: dict):
    qubits = {N: qcfg["qubits"]["noun"], S: qcfg["qubits"]["sentence"]}
    return ANSATZE[qcfg["ansatz"]](qubits, n_layers=qcfg["layers"])


def to_circuit(diagram: Diagram, ansatz, remove_cups: bool = True) -> quantum.Diagram:
    diagram = simplify(diagram)
    if remove_cups:
        diagram = _remove_cups(diagram)
    return ansatz(diagram)


def to_tensor(diagram: Diagram, dims: dict):
    # Cups are kept: lambeq's PytorchModel can't contract the cup-free form.
    ansatz = TensorAnsatz({N: Dim(dims["noun"]), S: Dim(dims["sentence"])})
    return ansatz(simplify(diagram))


def circuit_stats(circuit: quantum.Diagram) -> dict:
    width = max(len(layer.left) + max(len(layer.box.dom), len(layer.box.cod)) + len(layer.right)
                for layer in circuit.layers)
    return {
        "qubits": width,
        "gates": sum(not isinstance(b, (quantum.Ket, quantum.Bra)) for b in circuit.boxes),
        "postselected": sum(isinstance(b, quantum.Bra) for b in circuit.boxes),
        "parameters": len(circuit.free_symbols),
    }
