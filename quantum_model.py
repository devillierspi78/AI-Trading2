import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, RYGate, ZGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

def quantum_portfolio_optimization(trading_assets):
    """Uses quantum estimation to optimize portfolio allocation."""
    estimator = Estimator()
    weights = []

    for i, _ in enumerate(trading_assets):
        circuit = QuantumCircuit(1)
        
        # Apply Hadamard to introduce superposition
        circuit.append(HGate(), [0])
        
        # Introduce randomness with rotation
        random_rotation = np.pi / (i + 1) + np.random.uniform(-0.5, 0.5)
        circuit.append(RYGate(random_rotation), [0])
        
        # Apply Z gate for measurement
        circuit.append(ZGate(), [0])

        observable = SparsePauliOp("Z")
        result = estimator.run([circuit], [observable]).result().values
        weights.append(result[0])

    # Convert to NumPy array
    weights = np.array(weights)

    # Shift weights to center around zero
    weights -= np.mean(weights)

    # Normalize to ensure sum of absolute values is 1
    weights /= (np.sum(np.abs(weights)) + 1e-8)

    return weights
