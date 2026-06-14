import pennylane as qml
from models.quantum_circuit import q_node, n_qubits, n_layers

weight_shapes = {"weights": (n_layers, n_qubits)}


def get_quantum_layer():
    # Returns a quantum TorchLayer.
    return qml.qnn.TorchLayer(q_node, weight_shapes)
