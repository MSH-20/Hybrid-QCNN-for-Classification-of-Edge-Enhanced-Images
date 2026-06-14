import pennylane as qml
import numpy as np

n_qubits = 4   # one qubit per pixel in a 2x2 patch
n_layers = 2   # entangling layer depth for BasicEntanglerLayers
dev = qml.device("lightning.qubit", wires=n_qubits)


def quantum_conv_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


@qml.qnode(dev, interface='torch')
def q_node(inputs, weights):
    return quantum_conv_circuit(inputs, weights)
