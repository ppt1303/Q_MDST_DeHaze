
import sys
sys.path.insert(0, "..")

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import configs_trainH2C_QMDST as c
from models import DSNet
from myutils.dataloader import Dataloader

from metrics.metrics import psnr_pt2np, ssim_pt


def evaluate(name, checkpoint, Ten, Tde, loader, device):

    encoder = DSNet.Encoder().to(device)

    encoder.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True
        ),
        strict=True
    )

    encoder.eval()

    haze_psnr = []
    haze_ssim = []

    raw_psnr = []
    raw_ssim = []

    clipped_psnr = []
    clipped_ssim = []

    output_min = float("inf")
    output_max = -float("inf")
    output_sum = 0.0
    output_count = 0
    outside_count = 0

    with torch.inference_mode():

        for batch in loader:

            haze = batch[0].to(device)
            clear = batch[1].to(device)

            # Input haze baseline
            haze_psnr.append(
                float(psnr_pt2np(haze, clear))
            )

            haze_ssim.append(
                float(ssim_pt(haze, clear).mean().item())
            )

            # Student output
            feat, x1, x2 = encoder(haze)
            output = Tde(feat, x1, x2)

            # Raw output
            raw_psnr.append(
                float(psnr_pt2np(output, clear))
            )

            raw_ssim.append(
                float(ssim_pt(output, clear).mean().item())
            )

            # Output range
            output_min = min(
                output_min, output.min().item()
            )

            output_max = max(
                output_max, output.max().item()
            )

            output_sum += output.sum().item()
            output_count += output.numel()

            outside_count += (
                ((output < 0) | (output > 1))
                .sum().item()
            )

            # Clipped output
            clipped = output.clamp(0, 1)

            clipped_psnr.append(
                float(psnr_pt2np(clipped, clear))
            )

            clipped_ssim.append(
                float(ssim_pt(clipped, clear).mean().item())
            )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        "Hazy baseline PSNR:",
        round(np.mean(haze_psnr), 5)
    )

    print(
        "Hazy baseline SSIM:",
        round(np.mean(haze_ssim), 6)
    )

    print(
        "Raw output PSNR:",
        round(np.mean(raw_psnr), 5)
    )

    print(
        "Raw output SSIM:",
        round(np.mean(raw_ssim), 6)
    )

    print(
        "Clipped output PSNR:",
        round(np.mean(clipped_psnr), 5)
    )

    print(
        "Clipped output SSIM:",
        round(np.mean(clipped_ssim), 6)
    )

    print("Output min:", output_min)
    print("Output max:", output_max)

    print(
        "Output mean:",
        output_sum / output_count
    )

    print(
        "Pixels outside [0,1]:",
        round(100 * outside_count / output_count, 4),
        "%"
    )


def main():

    device = torch.device(c.device)

    print("=" * 60)
    print("Q-MDST H2C DIAGNOSTIC AUDIT")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))

    # Validation dataset
    dataset = Dataloader(
        c.haze_val_dir,
        c.clear_val_dir,
        transform=c.transforms_train,
        unaligned=False,
        model="val"
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Frozen Teacher Decoder
    Tde = DSNet.Decoder().to(device)

    Tde.load_state_dict(
        torch.load(
            c.saved_model_Tde_path,
            map_location=device,
            weights_only=True
        ),
        strict=True
    )

    Tde.eval()

    for p in Tde.parameters():
        p.requires_grad_(False)

    checkpoint_dir = (
        "../results/checkpoints/"
        "QMDST_Research_v1/H2C_Classical"
    )

    checkpoints = [
        (
            "BEST PSNR",
            checkpoint_dir + "/Sen_Best_PSNR.pth"
        ),
        (
            "BEST SSIM",
            checkpoint_dir + "/Sen_Best_SSIM.pth"
        ),
        (
            "LAST EPOCH",
            checkpoint_dir + "/Sen_last.pth"
        )
    ]

    for name, checkpoint in checkpoints:

        evaluate(
            name,
            checkpoint,
            None,
            Tde,
            loader,
            device
        )

    print("\n" + "=" * 60)
    print("H2C DIAGNOSTIC AUDIT COMPLETED")
    print("No checkpoint modified.")
    print("No training performed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
