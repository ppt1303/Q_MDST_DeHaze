
import sys
sys.path.insert(0, "..")

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from configs import configs_trainC2C_QMDST as configs
from models import DSNet
from metrics.losses import PerceptualLoss
from myutils.dataloader import Dataloader


def main():

    device = torch.device(configs.device)

    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is not available!")

    print("=" * 55)
    print("Q-MDST RESEARCH - C2C SMOKE TEST")
    print("=" * 55)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)

    torch.cuda.empty_cache()

    # ======================================
    # 1. DATA
    # ======================================

    dataset = Dataloader(
        haze_dir=configs.clear_dir,
        clear_dir=configs.clear_dir,
        transform=configs.transforms_train,
        unaligned=False
    )

    loader = DataLoader(
        dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=0
    )

    batch = next(iter(loader))

    x = batch[0].to(device)

    print("\n[1] DATA")
    print("Dataset length:", len(dataset))
    print("Input:", tuple(x.shape))
    print("Image:", batch[2][0])

    # ======================================
    # 2. INITIALIZE TEACHER
    # ======================================

    print("\n[2] INITIALIZING MODELS")

    Ten = DSNet.Encoder().to(device)
    Tde = DSNet.Decoder().to(device)

    Ten.train()
    Tde.train()

    params_Ten = sum(
        p.numel() for p in Ten.parameters()
        if p.requires_grad
    )

    params_Tde = sum(
        p.numel() for p in Tde.parameters()
        if p.requires_grad
    )

    print("Teacher Encoder parameters:", params_Ten)
    print("Shared Decoder parameters:", params_Tde)
    print("Total parameters:", params_Ten + params_Tde)

    # ======================================
    # 3. PERCEPTUAL LOSS
    # ======================================

    print("\n[3] INITIALIZING LOSSES")

    loss_l1 = nn.L1Loss().to(device)
    loss_per = PerceptualLoss().to(device)

    opt_Ten = optim.Adam(
        Ten.parameters(),
        lr=configs.lr,
        betas=(0.5, 0.999)
    )

    opt_Tde = optim.Adam(
        Tde.parameters(),
        lr=configs.lr,
        betas=(0.5, 0.999)
    )

    print("L1 Loss: initialized")
    print("VGG19 Perceptual Loss: initialized")

    # ======================================
    # 4. FORWARD
    # ======================================

    print("\n[4] TEACHER FORWARD")

    feat, x1, x2 = Ten(x)

    output = Tde(feat, x1, x2)

    print("Latent feature:", tuple(feat.shape))
    print("Skip feature 1:", tuple(x1.shape))
    print("Skip feature 2:", tuple(x2.shape))
    print("Output:", tuple(output.shape))

    assert output.shape == x.shape, (
        f"Output mismatch: {output.shape} vs {x.shape}"
    )

    print("Forward: PASSED")

    # ======================================
    # 5. COMPUTE LOSSES
    # ======================================

    print("\n[5] COMPUTING LOSSES")

    loss_L1 = (
        loss_l1(output, x) * configs.l1_weight
    )

    loss_Per = (
        loss_per(output, x) * configs.perceptual_weight
    )

    total_loss = loss_L1 + loss_Per

    losses = {
        "L1": loss_L1,
        "Perceptual": loss_Per,
        "Total": total_loss
    }

    for name, value in losses.items():

        if not torch.isfinite(value).all():
            raise RuntimeError(
                f"Non-finite loss: {name}"
            )

        print(f"{name}: {value.item():.6f}")

    # ======================================
    # 6. BACKWARD
    # ======================================

    print("\n[6] TEACHER BACKWARD")

    opt_Ten.zero_grad(set_to_none=True)
    opt_Tde.zero_grad(set_to_none=True)

    total_loss.backward()

    grad_Ten = sum(
        p.grad.detach().abs().sum().item()
        for p in Ten.parameters()
        if p.grad is not None
    )

    grad_Tde = sum(
        p.grad.detach().abs().sum().item()
        for p in Tde.parameters()
        if p.grad is not None
    )

    if grad_Ten == 0 or grad_Tde == 0:
        raise RuntimeError("Missing gradients!")

    opt_Ten.step()
    opt_Tde.step()

    print("Encoder gradient:", grad_Ten)
    print("Decoder gradient:", grad_Tde)
    print("Encoder optimizer: PASSED")
    print("Decoder optimizer: PASSED")

    # ======================================
    # 7. GPU MEMORY
    # ======================================

    torch.cuda.synchronize()

    peak_allocated = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved() / 1024**3
    )

    print("\n[7] GPU MEMORY")
    print(f"Peak allocated: {peak_allocated:.2f} GB")
    print(f"Peak reserved: {peak_reserved:.2f} GB")

    # ======================================
    # 8. FINAL
    # ======================================

    print("\n" + "=" * 55)
    print("C2C SMOKE TEST: PASSED")
    print("No checkpoint was saved or overwritten.")
    print("=" * 55)


if __name__ == "__main__":
    main()
