
import sys
sys.path.append("..")

import csv
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

from configs import configs_trainC2C_QMDST as configs
from models import DSNet
from myutils.dataloader import Dataloader
from myutils.utils_visual import LambdaLR
from metrics.losses import PerceptualLoss
from metrics.metrics import psnr, ssim


def load_validation_manifest():

    rows = []

    with open(
        configs.val_manifest_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            if row["split"] == "val":
                rows.append(row)

    if len(rows) != 175:
        raise RuntimeError(
            f"Expected 175 validation patches, got {len(rows)}"
        )

    return rows


def validate(Ten, Tde, loader, manifest, device):

    Ten.eval()
    Tde.eval()

    patch_map = {
        row["patch"]: row
        for row in manifest
    }

    canvases = {}
    weights = {}

    with torch.inference_mode():

        for batch in loader:

            x = batch[0].to(device)
            patch_name = batch[2][0]

            if patch_name not in patch_map:
                raise RuntimeError(
                    f"Patch missing from manifest: {patch_name}"
                )

            row = patch_map[patch_name]

            feat, x1, x2 = Ten(x)
            output = Tde(feat, x1, x2)

            prediction = (
                output[0].detach()
                .float().cpu().clamp(0, 1)
            )

            source = row["source"]

            x0 = int(row["x"])
            y0 = int(row["y"])
            width = int(row["width"])
            height = int(row["height"])

            if source not in canvases:

                full_path = (
                    Path(configs.val_full_clear_dir)
                    / source
                )

                with Image.open(full_path) as image:
                    gt = image.convert("RGB")
                    W, H = gt.size

                canvases[source] = torch.zeros(
                    (3, H, W),
                    dtype=torch.float32
                )

                weights[source] = torch.zeros(
                    (1, H, W),
                    dtype=torch.float32
                )

            canvases[source][
                :, y0:y0+height, x0:x0+width
            ] += prediction

            weights[source][
                :, y0:y0+height, x0:x0+width
            ] += 1.0

    scene_results = {}
    reconstructed = {}

    for source in sorted(canvases):

        count = weights[source]

        if torch.any(count == 0):
            raise RuntimeError(
                f"Uncovered pixels in validation image: {source}"
            )

        prediction = (
            canvases[source] / count
        ).clamp(0, 1)

        full_path = (
            Path(configs.val_full_clear_dir)
            / source
        )

        with Image.open(full_path) as image:
            target = np.asarray(
                image.convert("RGB"),
                dtype=np.float32
            ) / 255.0

        pred_np = (
            prediction.permute(1, 2, 0).numpy()
        )

        scene_psnr = float(psnr(target, pred_np))
        scene_ssim = float(ssim(target, pred_np))

        scene_results[source] = {
            "PSNR": scene_psnr,
            "SSIM": scene_ssim
        }

        reconstructed[source] = prediction

    mean_psnr = float(np.mean([
        value["PSNR"]
        for value in scene_results.values()
    ]))

    mean_ssim = float(np.mean([
        value["SSIM"]
        for value in scene_results.values()
    ]))

    Ten.train()
    Tde.train()

    return mean_psnr, mean_ssim, scene_results, reconstructed


def main():

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    torch.backends.cudnn.benchmark = True

    device = torch.device(configs.device)

    if device.type != "cuda":
        raise RuntimeError("CUDA is not available!")

    print("=" * 60)
    print("Q-MDST RESEARCH - C2C TEACHER TRAINING")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)

    # =========================================
    # 1. OUTPUT DIRECTORIES
    # =========================================

    model_dir = Path(configs.save_model_dir)
    log_dir = Path(configs.save_log_dir)
    image_dir = Path(configs.save_image_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    protected_files = [
        "Ten.pth",
        "Tde.pth",
        "Ten_Best_SSIM.pth",
        "Tde_Best_SSIM.pth",
        "Ten_last.pth",
        "Tde_last.pth",
        "C2C_training_state_last.pth"
    ]

    existing = [
        name for name in protected_files
        if (model_dir / name).exists()
    ]

    if existing:
        raise FileExistsError(
            "Existing C2C checkpoints detected. "
            "Back up or rename them before starting a new run: "
            + ", ".join(existing)
        )

    metrics_path = log_dir / "C2C_metrics.csv"

    if metrics_path.exists():
        raise FileExistsError(
            f"Existing metrics file: {metrics_path}"
        )

    # =========================================
    # 2. TRAIN DATA
    # =========================================

    train_dataset = Dataloader(
        haze_dir=configs.clear_dir,
        clear_dir=configs.clear_dir,
        transform=configs.transforms_train,
        unaligned=False,
        model="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=configs.n_threads,
        pin_memory=True
    )

    if len(train_dataset) != 180:
        raise RuntimeError(
            f"Expected 180 train patches, got {len(train_dataset)}"
        )

    # =========================================
    # 3. VALIDATION DATA
    # =========================================

    manifest = load_validation_manifest()

    val_dataset = Dataloader(
        haze_dir=configs.val_clear_dir,
        clear_dir=configs.val_clear_dir,
        transform=configs.transforms_train,
        unaligned=False,
        model="validation"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    if len(val_dataset) != 175:
        raise RuntimeError("Validation dataset count mismatch!")

    print("\n[1] DATA")
    print("Training patches:", len(train_dataset))
    print("Validation patches:", len(val_dataset))
    print(
        "Validation scenes:",
        len(set(row["source"] for row in manifest))
    )

    # =========================================
    # 4. INITIALIZE TEACHER
    # =========================================

    Ten = DSNet.Encoder().to(device)
    Tde = DSNet.Decoder().to(device)

    print("\n[2] TEACHER INITIALIZED")

    print(
        "Encoder parameters:",
        sum(p.numel() for p in Ten.parameters())
    )

    print(
        "Decoder parameters:",
        sum(p.numel() for p in Tde.parameters())
    )

    # =========================================
    # 5. LOSS AND OPTIMIZER
    # =========================================

    loss_l1 = nn.L1Loss().to(device)
    loss_per = PerceptualLoss().to(device)
    loss_per.eval()

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

    scheduler_Ten = torch.optim.lr_scheduler.LambdaLR(
        opt_Ten,
        lr_lambda=LambdaLR(
            configs.num_epoch,
            configs.start_epoch,
            configs.decay_epoch
        ).step
    )

    scheduler_Tde = torch.optim.lr_scheduler.LambdaLR(
        opt_Tde,
        lr_lambda=LambdaLR(
            configs.num_epoch,
            configs.start_epoch,
            configs.decay_epoch
        ).step
    )

    # =========================================
    # 6. TRAINING
    # =========================================

    best_psnr = -float("inf")
    best_ssim = -float("inf")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    columns = [
        "epoch",
        "train_l1",
        "train_perceptual",
        "train_total",
        "val_psnr_full_scene",
        "val_ssim_full_scene",
        "learning_rate",
        "time_seconds"
    ]

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=columns
        )

        writer.writeheader()

        for epoch in range(
            configs.start_epoch,
            configs.num_epoch + 1
        ):

            start_time = time.time()

            Ten.train()
            Tde.train()

            sum_l1 = 0.0
            sum_per = 0.0
            sum_total = 0.0

            for batch in train_loader:

                x = batch[0].to(
                    device,
                    non_blocking=True
                )

                feat, x1, x2 = Ten(x)
                output = Tde(feat, x1, x2)

                l1 = (
                    loss_l1(output, x)
                    * configs.l1_weight
                )

                perceptual = (
                    loss_per(output, x)
                    * configs.perceptual_weight
                )

                total = l1 + perceptual

                if not torch.isfinite(total):
                    raise RuntimeError(
                        f"Non-finite loss at epoch {epoch}"
                    )

                opt_Ten.zero_grad(set_to_none=True)
                opt_Tde.zero_grad(set_to_none=True)

                total.backward()

                opt_Ten.step()
                opt_Tde.step()

                sum_l1 += l1.item()
                sum_per += perceptual.item()
                sum_total += total.item()

            batches = len(train_loader)

            avg_l1 = sum_l1 / batches
            avg_per = sum_per / batches
            avg_total = sum_total / batches

            # =================================
            # VALIDATION
            # =================================

            val_psnr, val_ssim, scene_results, predictions = (
                validate(
                    Ten,
                    Tde,
                    val_loader,
                    manifest,
                    device
                )
            )

            # =================================
            # SAVE LAST CHECKPOINT
            # =================================

            torch.save(
                Ten.state_dict(),
                model_dir / "Ten_last.pth"
            )

            torch.save(
                Tde.state_dict(),
                model_dir / "Tde_last.pth"
            )

            # =================================
            # SAVE BEST PSNR
            # =================================

            if val_psnr > best_psnr:

                best_psnr = val_psnr

                torch.save(
                    Ten.state_dict(),
                    model_dir / "Ten.pth"
                )

                torch.save(
                    Tde.state_dict(),
                    model_dir / "Tde.pth"
                )

                best_dir = image_dir / "best_val_psnr"
                best_dir.mkdir(parents=True, exist_ok=True)

                for source, prediction in predictions.items():

                    save_image(
                        prediction,
                        best_dir / source
                    )

            # =================================
            # SAVE BEST SSIM
            # =================================

            if val_ssim > best_ssim:

                best_ssim = val_ssim

                torch.save(
                    Ten.state_dict(),
                    model_dir / "Ten_Best_SSIM.pth"
                )

                torch.save(
                    Tde.state_dict(),
                    model_dir / "Tde_Best_SSIM.pth"
                )

            # =================================
            # SAVE TRAINING STATE
            # =================================

            scheduler_Ten.step()
            scheduler_Tde.step()

            torch.save(
                {
                    "epoch": epoch,
                    "Ten": Ten.state_dict(),
                    "Tde": Tde.state_dict(),
                    "optimizer_Ten": opt_Ten.state_dict(),
                    "optimizer_Tde": opt_Tde.state_dict(),
                    "scheduler_Ten": scheduler_Ten.state_dict(),
                    "scheduler_Tde": scheduler_Tde.state_dict(),
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim
                },
                model_dir / "C2C_training_state_last.pth"
            )

            elapsed = time.time() - start_time

            current_lr = opt_Ten.param_groups[0]["lr"]

            writer.writerow({
                "epoch": epoch,
                "train_l1": avg_l1,
                "train_perceptual": avg_per,
                "train_total": avg_total,
                "val_psnr_full_scene": val_psnr,
                "val_ssim_full_scene": val_ssim,
                "learning_rate": current_lr,
                "time_seconds": elapsed
            })

            csv_file.flush()

            print(
                f"Epoch [{epoch:03d}/{configs.num_epoch:03d}] "
                f"| L1: {avg_l1:.6f} "
                f"| Perceptual: {avg_per:.6f} "
                f"| Total: {avg_total:.6f} "
                f"| Val PSNR: {val_psnr:.4f} "
                f"| Val SSIM: {val_ssim:.5f} "
                f"| Best PSNR: {best_psnr:.4f} "
                f"| Best SSIM: {best_ssim:.5f} "
                f"| Time: {elapsed:.1f}s"
            )

            for source, values in scene_results.items():
                print(
                    f"    {source}: "
                    f"PSNR={values['PSNR']:.4f}, "
                    f"SSIM={values['SSIM']:.5f}"
                )

    # =========================================
    # 7. FINAL REPORT
    # =========================================

    report = {
        "experiment": configs.category,
        "epochs": configs.num_epoch,
        "training_patches": len(train_dataset),
        "validation_patches": len(val_dataset),
        "validation_scenes": 5,
        "best_validation_psnr": best_psnr,
        "best_validation_ssim": best_ssim,
        "best_psnr_checkpoint": "Ten.pth / Tde.pth",
        "best_ssim_checkpoint": (
            "Ten_Best_SSIM.pth / Tde_Best_SSIM.pth"
        )
    }

    with open(
        log_dir / "C2C_final_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 60)
    print("C2C TRAINING COMPLETED")
    print("Best validation PSNR:", best_psnr)
    print("Best validation SSIM:", best_ssim)
    print("Checkpoints:", model_dir.resolve())
    print("Metrics:", metrics_path.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()
