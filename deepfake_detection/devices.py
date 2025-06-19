#!/usr/bin/python3
'''Contains utility code such as loggers, file opening operations, etc.
'''
import torch
import pennylane as qml
import numpy as np


class ClassicDevice:
    '''Create and return torch device.

    Args:
        gpus (int, optional): Use GPUs if specified. Defaults to 0.

    Returns:
        device (obj): torch device
    '''
    def __init__(self, gpus=0):
        device_type = "cpu"

        if torch.cuda.is_available() and gpus > 0:
            device_type = "cuda:0"

        self.device = torch.device(device_type)


class QuantumDevice:
    def __init__(self, qubit_type, number_qubits, number_layers, type_layers, diff_type):
        self.qubit_type = qubit_type
        self.number_qubits = number_qubits
        self.number_layers = number_layers
        self.type_layers = type_layers
        self.diff_type = diff_type
        self.device = qml.device(self.qubit_type, wires=self.number_qubits)

    def weights(self):
        weight_shapes = {}

        if self.type_layers == "basic":
            weight_shapes = {"weights": (self.number_layers, self.number_qubits)}

        if self.type_layers == "strong":
            weight_shapes = {"weights": (self.number_layers, self.number_qubits, 3)}

        return weight_shapes

    def params(self, random=False):
        params = ""

        if random:
            params = np.random.uniform(
                high=2 * np.pi,
                size=(self.number_layers, self.number_qubits)
            )
        else:
            if self.type_layers == 'basic':
                params = torch.nn.Parameter(torch.rand(
                    (self.number_layers, self.number_qubits),
                    requires_grad=True
                ))

            if self.type_layers == "strong":
                params = torch.nn.Parameter(torch.rand(
                    (self.number_layers, self.number_qubits, 3),
                    requires_grad=True
                ))

        return params

    def circuit(self, inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(self.number_qubits))

        if self.type_layers == "basic":
            qml.BasicEntanglerLayers(weights, wires=range(self.number_qubits))
        elif self.type_layers == "strong":
            qml.StronglyEntanglingLayers(weights, wires=range(self.number_qubits))

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.number_qubits)]

    def random_circuit(self, phi):
        for j in range(self.number_qubits):
            qml.RY(np.pi * phi[j], wires=j)

        qml.RandomLayers(
            self.params(random=True), wires=list(range(self.number_qubits))
        )

        return [qml.expval(qml.PauliZ(j)) for j in range(self.number_qubits)]
