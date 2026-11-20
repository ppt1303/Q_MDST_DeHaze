
import torch
import torch.nn as nn

from . import DSNet
from .QuantumLayer import QuantumChannelModulation


class QDSNetEncoder(DSNet.Encoder):

    """
    Quantum-Classical Hybrid DSNet Encoder.

    Original DSNet Encoder:
        124 state_dict entries.

    Quantum extension:
        4 qubits.
        2 variational layers.
        Residual channel modulation.

    Original Skip Connections are preserved.
    """

    def __init__(
        self,
        in_channels=3,
        base_dim=64,
        n_qubits=4,
        n_layers=2
    ):

        super().__init__(
            in_channels=in_channels,
            base_dim=base_dim
        )

        self.quantum_modulation = (
            QuantumChannelModulation(
                channels=base_dim,
                n_qubits=n_qubits,
                n_layers=n_layers
            )
        )

    def forward(self, x):

        # Original DSNet Encoder
        feat, x1, x2 = super().forward(x)

        # Quantum modulation at bottleneck
        feat = self.quantum_modulation(feat)

        # Preserve original output interface
        return feat, x1, x2
