
import sys
import time

sys.path.insert(0, "..")

import torch
import pennylane as qml

from models.QuantumLayer import QuantumChannelModulation


def main():

    torch.manual_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable!")

    device = torch.device("cuda")

    print("=" * 60)
    print("Q-MDST QUANTUM LAYER SMOKE TEST")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("PennyLane:", qml.__version__)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # =================================
    # 1. INITIALIZE
    # =================================

    print("\n[1] INITIALIZING QUANTUM LAYER")

    model = QuantumChannelModulation(
        channels=64,
        n_qubits=4,
        n_layers=2
    ).to(device)

    print("Qubits:", model.n_qubits)
    print("Variational layers:", model.n_layers)
    print(
        "Quantum parameters:",
        model.quantum.weights.numel()
    )

    # =================================
    # 2. INPUT
    # =================================

    print("\n[2] INPUT")

    feat = torch.randn(
        1, 64, 64, 64,
        device=device,
        requires_grad=True
    )

    print("Feature:", tuple(feat.shape))

    # =================================
    # 3. FORWARD
    # =================================

    print("\n[3] QUANTUM FORWARD")

    start = time.perf_counter()

    output = model(feat)

    torch.cuda.synchronize()

    elapsed_forward = (
        time.perf_counter() - start
    )

    print("Output:", tuple(output.shape))
    print(
        "Forward time:",
        round(elapsed_forward, 4),
        "seconds"
    )

    assert output.shape == feat.shape

    assert torch.isfinite(output).all()

    print("Forward: PASSED")

    # =================================
    # 4. BACKWARD
    # =================================

    print("\n[4] QUANTUM BACKWARD")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    optimizer.zero_grad(set_to_none=True)

    loss = output.square().mean()

    loss.backward()

    torch.cuda.synchronize()

    q_grad = model.quantum.weights.grad

    pre_grad = model.pre_quantum.weight.grad

    post_grad = model.post_quantum.weight.grad

    input_grad = feat.grad

    assert q_grad is not None
    assert pre_grad is not None
    assert post_grad is not None
    assert input_grad is not None

    assert torch.isfinite(q_grad).all()
    assert torch.isfinite(pre_grad).all()
    assert torch.isfinite(post_grad).all()
    assert torch.isfinite(input_grad).all()

    print(
        "Quantum gradient:",
        q_grad.abs().sum().item()
    )

    print(
        "Pre-quantum gradient:",
        pre_grad.abs().sum().item()
    )

    print(
        "Post-quantum gradient:",
        post_grad.abs().sum().item()
    )

    print(
        "Input gradient:",
        input_grad.abs().sum().item()
    )

    if q_grad.abs().sum().item() == 0:
        raise RuntimeError(
            "Quantum gradient is zero!"
        )

    optimizer.step()

    print("Optimizer step: PASSED")

    # =================================
    # 5. GPU MEMORY
    # =================================

    torch.cuda.synchronize()

    allocated = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    print("\n[5] GPU MEMORY")

    print(
        f"Peak allocated: {allocated:.3f} GB"
    )

    print(
        f"Peak reserved: {reserved:.3f} GB"
    )

    # =================================
    # 6. FINAL
    # =================================

    print("\n" + "=" * 60)
    print("QUANTUM LAYER SMOKE TEST: PASSED")
    print("Quantum forward: PASSED")
    print("Quantum backward: PASSED")
    print("No checkpoint modified.")
    print("No training performed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
