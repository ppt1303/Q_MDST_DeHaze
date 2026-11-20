
import sys
sys.path.insert(0, "..")

from pathlib import Path

import torch
import torch.nn as nn

from configs import configs_trainH2C_QMDST as c
from models import DSNet
from models.QDSNet import QDSNetEncoder
from myutils.dataloader import Dataloader_TrainH2C


def main():

    device = torch.device("cuda")

    ROOT = Path(__file__).resolve().parents[1]

    checkpoint_dir = (
        ROOT / "results" /
        "checkpoints" /
        "QMDST_Research_v1"
    )

    classical_path = (
        checkpoint_dir /
        "H2C_Classical" /
        "Sen_Best_PSNR.pth"
    )

    teacher_path = checkpoint_dir / "Tde.pth"

    print("=" * 65)
    print("Q-MDST - HYBRID ENCODER INTEGRATION TEST")
    print("=" * 65)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)

    # =====================================
    # 1. DATA
    # =====================================

    dataset = Dataloader_TrainH2C(
        c.haze_dir,
        c.clear_dir,
        c.stylized_dir,
        transform=c.transforms_train
    )

    haze, clear, *_ = dataset[0]

    haze = haze.unsqueeze(0).to(device)
    clear = clear.unsqueeze(0).to(device)

    print("\n[1] DATA")
    print("Haze:", tuple(haze.shape))
    print("Clear:", tuple(clear.shape))

    # =====================================
    # 2. INITIALIZE QDSNET
    # =====================================

    print("\n[2] INITIALIZING QDSNET")

    encoder = QDSNetEncoder(
        base_dim=64,
        n_qubits=4,
        n_layers=2
    ).to(device)

    # =====================================
    # 3. LOAD CLASSICAL CHECKPOINT
    # =====================================

    print("\n[3] LOADING CLASSICAL CHECKPOINT")

    classical_state = torch.load(
        classical_path,
        map_location="cpu",
        weights_only=True
    )

    result = encoder.load_state_dict(
        classical_state,
        strict=False
    )

    expected_missing = {
        key for key in encoder.state_dict()
        if key.startswith("quantum_modulation.")
    }

    actual_missing = set(result.missing_keys)

    print("Classical keys:", len(classical_state))
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)

    assert len(classical_state) == 124

    assert actual_missing == expected_missing

    assert len(result.unexpected_keys) == 0

    # Verify original Classical weights
    current_state = encoder.state_dict()

    for key, original in classical_state.items():

        loaded = current_state[key].detach().cpu()

        assert torch.equal(
            original.cpu(),
            loaded
        ), f"Classical weight mismatch: {key}"

    print("Classical weights: EXACT MATCH")
    print("Quantum parameters: NEW INITIALIZATION")
    print("Checkpoint loading: PASSED")

    # =====================================
    # 4. VERIFY PARAMETER DEVICES
    # =====================================

    print("\n[4] DEVICE VERIFICATION")

    print(
        "Classical:",
        next(encoder.conv_init1.parameters()).device
    )

    print(
        "Quantum:",
        encoder.quantum_modulation.quantum.weights.device
    )

    print(
        "Pre-quantum:",
        next(
            encoder.quantum_modulation.pre_quantum.parameters()
        ).device
    )

    print(
        "Post-quantum:",
        next(
            encoder.quantum_modulation.post_quantum.parameters()
        ).device
    )

    # =====================================
    # 5. LOAD TEACHER DECODER
    # =====================================

    print("\n[5] LOADING TEACHER DECODER")

    decoder = DSNet.Decoder().to(device)

    decoder.load_state_dict(
        torch.load(
            teacher_path,
            map_location="cpu",
            weights_only=True
        ),
        strict=True
    )

    decoder.eval()

    for p in decoder.parameters():
        p.requires_grad_(False)

    print("Teacher Decoder: STRICT LOAD PASSED")
    print("Teacher Decoder: FROZEN")

    # =====================================
    # 6. HYBRID FORWARD
    # =====================================

    print("\n[6] HYBRID FORWARD")

    encoder.train()

    feat, x1, x2 = encoder(haze)

    output = decoder(feat, x1, x2)

    print("Latent:", tuple(feat.shape))
    print("Skip 1:", tuple(x1.shape))
    print("Skip 2:", tuple(x2.shape))
    print("Output:", tuple(output.shape))

    assert feat.shape == (1, 64, 64, 64)
    assert x1.shape == (1, 64, 256, 256)
    assert x2.shape == (1, 64, 128, 128)
    assert output.shape == (1, 3, 256, 256)

    assert torch.isfinite(output).all()

    print("Hybrid forward: PASSED")

    # =====================================
    # 7. HYBRID BACKWARD
    # =====================================

    print("\n[7] HYBRID BACKWARD")

    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=1e-4
    )

    optimizer.zero_grad(set_to_none=True)

    loss = nn.L1Loss()(output, clear)

    loss.backward()

    qmodule = encoder.quantum_modulation

    gradient_checks = {
        "Quantum": qmodule.quantum.weights.grad,
        "Pre-quantum": qmodule.pre_quantum.weight.grad,
        "Post-quantum": qmodule.post_quantum.weight.grad,
        "Classical Encoder": encoder.conv_init1.weight.grad
    }

    for name, grad in gradient_checks.items():

        if grad is None:
            raise RuntimeError(
                f"{name} gradient is None!"
            )

        if not torch.isfinite(grad).all():
            raise RuntimeError(
                f"{name} gradient is invalid!"
            )

        magnitude = grad.abs().sum().item()

        print(f"{name} gradient: {magnitude:.8f}")

        if magnitude == 0:
            raise RuntimeError(
                f"{name} gradient is zero!"
            )

    assert all(
        p.grad is None
        for p in decoder.parameters()
    )

    optimizer.step()

    print("Hybrid optimizer: PASSED")
    print("Teacher weights: FROZEN")

    # =====================================
    # 8. PARAMETER COUNT
    # =====================================

    print("\n[8] PARAMETERS")

    total = sum(
        p.numel() for p in encoder.parameters()
    )

    quantum = sum(
        p.numel()
        for p in qmodule.parameters()
    )

    print("Total QDSNet parameters:", total)
    print("New Quantum block parameters:", quantum)
    print("Quantum circuit parameters:", qmodule.quantum.weights.numel())

    # =====================================
    # 9. GPU MEMORY
    # =====================================

    torch.cuda.synchronize()

    print(
        "GPU allocated:",
        round(
            torch.cuda.memory_allocated() / 1024**3,
            3
        ),
        "GB"
    )

    # =====================================
    # 10. FINAL
    # =====================================

    print("\n" + "=" * 65)
    print("QDSNET INTEGRATION TEST: PASSED")
    print("124 Classical tensors preserved.")
    print("Quantum circuit integrated.")
    print("Teacher Decoder preserved.")
    print("No checkpoint saved or overwritten.")
    print("No training performed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
