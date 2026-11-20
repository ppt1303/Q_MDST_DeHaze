
import math

import torch
import torch.nn as nn
import pennylane as qml


class CPUQuantumCircuit(nn.Module):

    """
    PennyLane CPU quantum simulator.

    Quantum parameters remain on CPU even when
    the parent model is moved to CUDA.

    Gradients propagate through CPU/GPU copies.
    """

    def __init__(self, n_qubits=4, n_layers=2):

        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.dev = qml.device(
            "default.qubit",
            wires=n_qubits
        )

        @qml.qnode(
            self.dev,
            interface="torch",
            diff_method="backprop"
        )
        def circuit(inputs, weights):

            qml.AngleEmbedding(
                inputs,
                wires=range(n_qubits),
                rotation="Y"
            )

            qml.StronglyEntanglingLayers(
                weights,
                wires=range(n_qubits)
            )

            return tuple(
                qml.expval(qml.PauliZ(i))
                for i in range(n_qubits)
            )

        self.circuit = circuit

        self.weights = nn.Parameter(
            torch.empty(
                n_layers,
                n_qubits,
                3,
                device="cpu",
                dtype=torch.float32
            )
        )

        nn.init.normal_(
            self.weights,
            mean=0.0,
            std=0.05
        )

    def _apply(self, fn, recurse=True):

        # Keep the quantum circuit and its parameters
        # on CPU when the parent model calls .to(cuda).
        return self

    def forward(self, inputs):

        inputs_cpu = inputs.to(
            device="cpu",
            dtype=torch.float32
        )

        result = self.circuit(
            inputs_cpu,
            self.weights
        )

        if isinstance(result, (tuple, list)):
            result = torch.stack(
                list(result),
                dim=-1
            )

        return result


class QuantumChannelModulation(nn.Module):

    """
    Hybrid Quantum-Classical Channel Modulation.

    Input:
        [B, 64, H, W]

    Quantum:
        4 qubits
        2 variational layers

    Output:
        [B, 64, H, W]
    """

    def __init__(
        self,
        channels=64,
        n_qubits=4,
        n_layers=2
    ):

        super().__init__()

        self.channels = channels
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical feature extraction
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.pre_quantum = nn.Linear(
            channels,
            n_qubits
        )

        # Quantum circuit remains on CPU
        self.quantum = CPUQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers
        )

        # Classical feature reconstruction
        self.post_quantum = nn.Linear(
            n_qubits,
            channels
        )

        nn.init.normal_(
            self.post_quantum.weight,
            mean=0.0,
            std=0.02
        )

        nn.init.zeros_(
            self.post_quantum.bias
        )

        self.residual_scale = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(self, feat):

        if feat.ndim != 4:
            raise ValueError(
                "Expected input [B,C,H,W]"
            )

        if feat.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels"
            )

        batch_size = feat.shape[0]

        # 1. Global average pooling
        pooled = self.pool(feat).flatten(1)

        # 2. Classical encoding
        angles = (
            torch.tanh(
                self.pre_quantum(pooled)
            ) * math.pi
        )

        # 3. Quantum processing on CPU
        q_features = self.quantum(angles)

        # 4. Transfer quantum outputs to GPU
        q_features = q_features.to(
            device=feat.device,
            dtype=feat.dtype
        )

        # 5. Classical projection
        modulation = self.post_quantum(
            q_features
        )

        modulation = torch.tanh(
            modulation
        )

        modulation = modulation.reshape(
            batch_size,
            self.channels,
            1,
            1
        )

        # 6. Residual feature modulation
        output = feat * (
            1.0 +
            self.residual_scale * modulation
        )

        return output
