from lambeq import cups_reader, RemoveCupsRewriter, AtomicType, IQPAnsatz

def test_phase_2_with_mock_input():
    print("Generating mock diagram offline...")
    mock_diagram = cups_reader.sentence2diagram("Rook pins queen")

    # Phase 1: Rewriting
    rewriter = RemoveCupsRewriter()
    clean_diagram = rewriter(mock_diagram)

    # Phase 2: Quantum compilation
    type_map = {AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}
    ansatz = IQPAnsatz(type_map, n_layers=1, n_single_qubit_params=3)
    pqc = ansatz(clean_diagram)

    # Inspect parameters and qubit metrics
    symbols = [str(s) for s in pqc.free_symbols]
    output_qubits = len(pqc.cod)
    
    # Count total initialized qubits via state preparation (Ket) boxes
    total_qubits = len([box for box in pqc.boxes if 'Ket' in type(box).__name__ or 'Ket' in box.name])
    if total_qubits == 0:
        total_qubits = output_qubits  # Fallback to codomain wire count

    print("\n--- Audit Results ---")
    print(f"Output Qubit Width : {output_qubits} qubits")
    print(f"Total Qubits Used  : {total_qubits} qubits")
    print(f"Total Parameters   : {len(symbols)}")
    print(f"Sample Parameters  : {symbols[:6]}")
    print("---------------------")

    # Assertions
    assert output_qubits > 0 or total_qubits > 0, "Error: Circuit has no qubits."
    assert len(symbols) > 0, "Error: No trainable rotation parameters found."
    print("✅ Phase 2 compilation test PASSED!")

if __name__ == "__main__":
    test_phase_2_with_mock_input()
