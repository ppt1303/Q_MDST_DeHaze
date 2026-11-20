
import sys
sys.path.insert(0, "..")

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from configs import configs_trainH2C_QMDST as c
from models import DSNet

from myutils.dataloader import Dataloader_TrainH2C

from metrics.losses import (
    PerceptualLoss,
    ContrastLoss,
    calc_self_perceptual_loss
)


def curriculum_weight(difficulty):

    diff_list = [18, 20, 25, 27, 32]

    weights = [
        (1 + c.cl_lambda) if difficulty > x
        else (1 - c.cl_lambda)
        for x in diff_list
    ]

    weights.append(len(diff_list))

    return [v / sum(weights) for v in weights]


def main():

    device = torch.device(c.device)

    if device.type != "cuda":
        raise RuntimeError("CUDA is unavailable!")

    print("=" * 60)
    print("Q-MDST RESEARCH - H2C CLASSICAL SMOKE TEST")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("PennyLane:", __import__("pennylane").__version__)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # =====================================
    # 1. LOAD DATA
    # =====================================

    dataset = Dataloader_TrainH2C(
        c.haze_dir,
        c.clear_dir,
        c.stylized_dir,
        transform=c.transforms_train
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    batch = next(iter(loader))

    haze = batch[0].to(device)
    clear = batch[1].to(device)

    candidate_list = batch[3]
    candidate_list_a = batch[4]

    print("\n[1] DATA")
    print("Dataset:", len(dataset))
    print("Haze:", tuple(haze.shape))
    print("Clear:", tuple(clear.shape))
    print("Candidate groups:", len(candidate_list))
    print("Aligned groups:", len(candidate_list_a))

    for i, candidate in enumerate(candidate_list):
        print(f"Candidate {i+1}:", tuple(candidate.shape))

        assert candidate.shape == (1, 3, 256, 256)

    # =====================================
    # 2. INITIALIZE MODELS
    # =====================================

    print("\n[2] INITIALIZING MODELS")

    encoder = DSNet.Encoder().to(device)
    Ten = DSNet.Encoder().to(device)
    Tde = DSNet.Decoder().to(device)

    Ten.load_state_dict(
        torch.load(
            c.saved_model_Ten_path,
            map_location=device,
            weights_only=True
        ),
        strict=True
    )

    Tde.load_state_dict(
        torch.load(
            c.saved_model_Tde_path,
            map_location=device,
            weights_only=True
        ),
        strict=True
    )

    # Freeze pretrained Teacher
    Ten.eval()
    Tde.eval()

    for p in Ten.parameters():
        p.requires_grad_(False)

    for p in Tde.parameters():
        p.requires_grad_(False)

    encoder.train()

    print("Student Encoder: INITIALIZED")
    print("Teacher Encoder: STRICT LOAD PASSED")
    print("Teacher Decoder: STRICT LOAD PASSED")

    # =====================================
    # 3. INITIALIZE LOSSES
    # =====================================

    print("\n[3] INITIALIZING LOSSES")

    loss_l1 = nn.L1Loss().to(device)

    loss_per = PerceptualLoss().to(device)
    loss_cl1 = ContrastLoss().to(device)

    loss_per.eval()
    loss_cl1.eval()

    opt_encoder = optim.Adam(
        encoder.parameters(),
        lr=c.lr,
        betas=(0.5, 0.999)
    )

    weights = curriculum_weight(0)

    print("Curriculum weights:", weights)
    print("L1: initialized")
    print("Perceptual: initialized")
    print("Contrastive: initialized")

    # =====================================
    # 4. FORWARD
    # =====================================

    print("\n[4] STUDENT AND TEACHER FORWARD")

    with torch.no_grad():
        feat_gt, x1_gt, x2_gt = Ten(clear)

    feat, x1, x2 = encoder(haze)

    output = Tde(feat, x1, x2)

    print("Student latent:", tuple(feat.shape))
    print("Teacher latent:", tuple(feat_gt.shape))
    print("Student skip 1:", tuple(x1.shape))
    print("Student skip 2:", tuple(x2.shape))
    print("Output:", tuple(output.shape))

    assert output.shape == clear.shape

    print("Forward: PASSED")

    # =====================================
    # 5. COMPUTE ORIGINAL H2C LOSSES
    # =====================================

    print("\n[5] COMPUTING H2C LOSSES")

    loss_self = (
        calc_self_perceptual_loss(
            [x1, x2, feat],
            [x1_gt, x2_gt, feat_gt]
        ) * c.selfperceptual_weight
    )

    loss_l1_value = (
        loss_l1(output, clear) * c.l1_weight
    )

    loss_per_value = (
        loss_per(output, clear) * c.perceptual_weight
    )

    loss_cl = (
        loss_cl1(
            output,
            clear,
            haze,
            candidate_list,
            weights
        ) * c.cl1_weight
    )

    total_loss = (
        loss_self
        + loss_l1_value
        + loss_per_value
        + loss_cl
    )

    losses = {
        "Self_Perceptual": loss_self,
        "L1": loss_l1_value,
        "Perceptual": loss_per_value,
        "Contrastive": loss_cl,
        "Total": total_loss
    }

    for name, value in losses.items():

        if not torch.isfinite(value).all():
            raise RuntimeError(
                f"Invalid loss: {name}"
            )

        print(f"{name}: {value.item():.6f}")

    # =====================================
    # 6. BACKWARD
    # =====================================

    print("\n[6] STUDENT BACKWARD")

    opt_encoder.zero_grad(set_to_none=True)

    total_loss.backward()

    grad_sum = sum(
        p.grad.detach().abs().sum().item()
        for p in encoder.parameters()
        if p.grad is not None
    )

    if grad_sum <= 0:
        raise RuntimeError("Student gradient is zero!")

    opt_encoder.step()

    print("Student gradient:", grad_sum)
    print("Student optimizer: PASSED")

    assert all(
        p.grad is None
        for p in Ten.parameters()
    )

    assert all(
        p.grad is None
        for p in Tde.parameters()
    )

    print("Teacher weights: FROZEN")
    print("Teacher checkpoints: PRESERVED")

    # =====================================
    # 7. GPU MEMORY
    # =====================================

    torch.cuda.synchronize()

    allocated = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    reserved = (
        torch.cuda.max_memory_reserved() / 1024**3
    )

    print("\n[7] GPU MEMORY")
    print(f"Peak allocated: {allocated:.2f} GB")
    print(f"Peak reserved: {reserved:.2f} GB")

    # =====================================
    # 8. FINAL
    # =====================================

    print("\n" + "=" * 60)
    print("H2C CLASSICAL SMOKE TEST: PASSED")
    print("No checkpoint saved or overwritten.")
    print("No full training performed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
