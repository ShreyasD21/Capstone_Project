import torch
from lambeq import cups_reader, RemoveCupsRewriter, AtomicType, IQPAnsatz, PennyLaneModel

def test_full_pipeline():
    print("--- Testing Full Part 2 Pipeline (Quantum Engine) ---")
    
    # 1. Mock Input (Part 1 -> Phase 1)
    sentences = ["Rook pins queen", "White blunders knight"]
    diagrams = [cups_reader.sentence2diagram(s) for s in sentences]
    
    rewriter = RemoveCupsRewriter()
    clean_diagrams = [rewriter(d) for d in diagrams]
    
    # 2. Compile to Quantum Circuits (Phase 2)
    type_map = {AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}
    ansatz = IQPAnsatz(type_map, n_layers=1, n_single_qubit_params=3)
    circuits = [ansatz(d) for d in clean_diagrams]
    
    # 3. Model Wrapping & Initializing Weights (Phase 3)
    # PennyLaneModel executes quantum circuits as PyTorch autograd layers
    model = PennyLaneModel.from_diagrams(circuits, probabilities=True, normalize=True)
    model.initialise_weights()
    
    # Run forward pass through quantum backend
    outputs = model(circuits)
    
    print("\n--- Model Output Summary ---")
    print(f"Number of Sentences Processed : {len(outputs)}")
    print(f"Output Feature Shape         : {outputs.shape}")
    print(f"Learned Weight Count         : {len(model.weights)}")
    
    # 4. Mock Backpropagation Step
    dummy_labels = torch.tensor([1, 0], dtype=torch.long)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    loss = loss_fn(outputs, dummy_labels)
    loss.backward()
    optimizer.step()
    
    print(f"Sample Loss Step Value       : {loss.item():.4f}")
    print("✅ Full Pipeline Dry Run PASSED!")

if __name__ == "__main__":
    test_full_pipeline()
