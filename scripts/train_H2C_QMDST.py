
import sys
sys.path.insert(0, "..")

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

from configs import configs_trainH2C_QMDST as c
from models import DSNet

from myutils.dataloader import (
    Dataloader,
    Dataloader_TrainH2C
)

from myutils.utils_visual import LambdaLR

from metrics.losses import (
    PerceptualLoss,
    ContrastLoss,
    calc_self_perceptual_loss
)

from metrics.metrics import psnr, ssim


ROOT = Path(__file__).resolve().parents[1]


def curriculum_weight(difficulty):

    diff_list = [18, 20, 25, 27, 32]

    weights = [
        (1 + c.cl_lambda) if difficulty > x
        else (1 - c.cl_lambda)
        for x in diff_list
    ]

    weights.append(len(diff_list))

    return [v / sum(weights) for v in weights]


def load_manifest():

    path = (
        ROOT / "datasets" /
        "NH_HAZE_QMDST_v1_eval256" /
        "patch_manifest.csv"
    )

    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r["split"] == "val"]

    if len(rows) != 175:
        raise RuntimeError("Validation manifest mismatch!")

    return rows


def validate(encoder, Tde, loader, manifest, device):

    encoder.eval()
    Tde.eval()

    patch_map = {
        row["patch"]: row for row in manifest
    }

    canvases = {}
    counts = {}

    with torch.inference_mode():

        for batch in loader:

            haze = batch[0].to(device)

            patch_name = batch[3][0]

            if patch_name not in patch_map:
                raise RuntimeError(
                    f"Missing validation patch: {patch_name}"
                )

            row = patch_map[patch_name]

            feat, x1, x2 = encoder(haze)
            output = Tde(feat, x1, x2)

            prediction = (
                output[0].float().cpu().clamp(0, 1)
            )

            source = row["source"]

            x = int(row["x"])
            y = int(row["y"])
            w = int(row["width"])
            h = int(row["height"])

            if source not in canvases:

                full_path = (
                    ROOT / "datasets" /
                    "NH_HAZE_QMDST_v1" /
                    "val" / "clear" / source
                )

                with Image.open(full_path) as im:
                    target = im.convert("RGB")
                    W, H = target.size

                canvases[source] = torch.zeros(
                    (3, H, W), dtype=torch.float32
                )

                counts[source] = torch.zeros(
                    (1, H, W), dtype=torch.float32
                )

            canvases[source][:, y:y+h, x:x+w] += prediction
            counts[source][:, y:y+h, x:x+w] += 1

    results = {}
    predictions = {}

    for source in sorted(canvases):

        if torch.any(counts[source] == 0):
            raise RuntimeError(
                f"Uncovered pixels: {source}"
            )

        pred = (
            canvases[source] / counts[source]
        ).clamp(0, 1)

        full_path = (
            ROOT / "datasets" /
            "NH_HAZE_QMDST_v1" /
            "val" / "clear" / source
        )

        with Image.open(full_path) as im:
            target = (
                np.asarray(im.convert("RGB"), dtype=np.float32)
                / 255.0
            )

        pred_np = (
            pred.permute(1, 2, 0).numpy()
        )

        p = float(psnr(pred_np, target))
        s = float(ssim(pred_np, target))

        results[source] = {
            "PSNR": p,
            "SSIM": s
        }

        predictions[source] = pred

    mean_psnr = float(np.mean([
        v["PSNR"] for v in results.values()
    ]))

    mean_ssim = float(np.mean([
        v["SSIM"] for v in results.values()
    ]))

    encoder.train()

    return mean_psnr, mean_ssim, results, predictions


def main():

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    torch.backends.cudnn.benchmark = True

    device = torch.device(c.device)

    if device.type != "cuda":
        raise RuntimeError("CUDA is unavailable!")

    print("=" * 65)
    print("Q-MDST RESEARCH - H2C CLASSICAL TRAINING")
    print("=" * 65)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)

    # =====================================
    # 1. OUTPUT DIRECTORIES
    # =====================================

    model_dir = ROOT / "results" / "checkpoints" / c.category
    log_dir = ROOT / "results" / "logs" / c.category
    image_dir = ROOT / "results" / "dehaze" / c.category

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    protected = [
        "Sen.pth",
        "Sen_last.pth",
        "Sen_Best_PSNR.pth",
        "Sen_Best_SSIM.pth",
        "H2C_training_state_last.pth"
    ]

    existing = [
        name for name in protected
        if (model_dir / name).exists()
    ]

    if existing:
        raise FileExistsError(
            "Existing H2C checkpoints detected: "
            + ", ".join(existing)
        )

    metrics_path = log_dir / "H2C_metrics.csv"

    if metrics_path.exists():
        raise FileExistsError(
            f"Existing metrics file: {metrics_path}"
        )

    # =====================================
    # 2. TRAIN DATA
    # =====================================

    train_dataset = Dataloader_TrainH2C(
        c.haze_dir,
        c.clear_dir,
        c.stylized_dir,
        transform=c.transforms_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=c.batch_size,
        shuffle=True,
        num_workers=c.n_threads,
        pin_memory=True
    )

    if len(train_dataset) != 180:
        raise RuntimeError("Training dataset mismatch!")

    # =====================================
    # 3. VALIDATION DATA
    # =====================================

    val_dataset = Dataloader(
        c.haze_val_dir,
        c.clear_val_dir,
        transform=c.transforms_train,
        unaligned=False,
        model="validation"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    manifest = load_manifest()

    if len(val_dataset) != 175:
        raise RuntimeError("Validation dataset mismatch!")

    print("\n[1] DATA")
    print("Training:", len(train_dataset))
    print("Validation patches:", len(val_dataset))
    print("Validation scenes:", len(set(
        r["source"] for r in manifest
    )))

    # =====================================
    # 4. INITIALIZE MODELS
    # =====================================

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

    Ten.eval()
    Tde.eval()

    for p in Ten.parameters():
        p.requires_grad_(False)

    for p in Tde.parameters():
        p.requires_grad_(False)

    encoder.train()

    print("\n[2] MODELS")
    print("Student Encoder: initialized")
    print("Teacher Encoder: frozen")
    print("Teacher Decoder: frozen")

    # =====================================
    # 5. LOSSES AND OPTIMIZER
    # =====================================

    loss_l1 = nn.L1Loss().to(device)
    loss_per = PerceptualLoss().to(device)
    loss_cl = ContrastLoss().to(device)

    loss_per.eval()
    loss_cl.eval()

    optimizer = optim.Adam(
        encoder.parameters(),
        lr=c.lr,
        betas=(0.5, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=LambdaLR(
            c.num_epoch,
            c.start_epoch,
            c.decay_epoch
        ).step
    )

    # =====================================
    # 6. TRAINING
    # =====================================

    best_psnr = 0.0
    best_ssim = 0.0

    columns = [
        "epoch",
        "self_perceptual",
        "l1",
        "perceptual",
        "contrastive",
        "total",
        "val_psnr_full_scene",
        "val_ssim_full_scene",
        "learning_rate",
        "seconds"
    ]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for epoch in range(1, c.num_epoch + 1):

            start = time.time()

            encoder.train()

            weights = curriculum_weight(best_psnr)

            sums = {
                "self": 0.0,
                "l1": 0.0,
                "per": 0.0,
                "cl": 0.0,
                "total": 0.0
            }

            lr_used = optimizer.param_groups[0]["lr"]

            for batch in train_loader:

                haze = batch[0].to(
                    device, non_blocking=True
                )

                clear = batch[1].to(
                    device, non_blocking=True
                )

                candidate_list = batch[3]

                # Frozen Teacher
                with torch.no_grad():
                    feat_gt, x1_gt, x2_gt = Ten(clear)

                # Student
                feat, x1, x2 = encoder(haze)

                # Frozen Decoder, gradients still pass to Student
                output = Tde(feat, x1, x2)

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
                    loss_per(output, clear)
                    * c.perceptual_weight
                )

                loss_cl_value = (
                    loss_cl(
                        output,
                        clear,
                        haze,
                        candidate_list,
                        weights
                    ) * c.cl1_weight
                )

                total = (
                    loss_self
                    + loss_l1_value
                    + loss_per_value
                    + loss_cl_value
                )

                if not torch.isfinite(total):
                    raise RuntimeError(
                        f"Non-finite loss at epoch {epoch}"
                    )

                optimizer.zero_grad(set_to_none=True)

                total.backward()

                optimizer.step()

                sums["self"] += loss_self.item()
                sums["l1"] += loss_l1_value.item()
                sums["per"] += loss_per_value.item()
                sums["cl"] += loss_cl_value.item()
                sums["total"] += total.item()

            n = len(train_loader)

            for key in sums:
                sums[key] /= n

            # =================================
            # VALIDATION
            # =================================

            val_psnr, val_ssim, scene_results, predictions = (
                validate(
                    encoder,
                    Tde,
                    val_loader,
                    manifest,
                    device
                )
            )

            elapsed = time.time() - start

            # Save last checkpoint
            torch.save(
                encoder.state_dict(),
                model_dir / "Sen_last.pth"
            )

            # Best PSNR
            if val_psnr > best_psnr:

                best_psnr = val_psnr

                torch.save(
                    encoder.state_dict(),
                    model_dir / "Sen_Best_PSNR.pth"
                )

                out_dir = image_dir / "val_psnr"
                out_dir.mkdir(parents=True, exist_ok=True)

                for name, pred in predictions.items():
                    save_image(pred, out_dir / name)

            # Best SSIM
            if val_ssim > best_ssim:

                best_ssim = val_ssim

                torch.save(
                    encoder.state_dict(),
                    model_dir / "Sen_Best_SSIM.pth"
                )

                out_dir = image_dir / "val_ssim"
                out_dir.mkdir(parents=True, exist_ok=True)

                for name, pred in predictions.items():
                    save_image(pred, out_dir / name)

            scheduler.step()

            # Training state
            torch.save(
                {
                    "epoch": epoch,
                    "encoder": encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim
                },
                model_dir / "H2C_training_state_last.pth"
            )

            writer.writerow({
                "epoch": epoch,
                "self_perceptual": sums["self"],
                "l1": sums["l1"],
                "perceptual": sums["per"],
                "contrastive": sums["cl"],
                "total": sums["total"],
                "val_psnr_full_scene": val_psnr,
                "val_ssim_full_scene": val_ssim,
                "learning_rate": lr_used,
                "seconds": elapsed
            })

            f.flush()

            print(
                f"Epoch [{epoch:03d}/{c.num_epoch}] "
                f"| SelfPer: {sums['self']:.5f} "
                f"| L1: {sums['l1']:.5f} "
                f"| Per: {sums['per']:.5f} "
                f"| CL: {sums['cl']:.5f} "
                f"| Total: {sums['total']:.5f} "
                f"| PSNR: {val_psnr:.4f} "
                f"| SSIM: {val_ssim:.5f} "
                f"| Best PSNR: {best_psnr:.4f} "
                f"| Best SSIM: {best_ssim:.5f} "
                f"| Time: {elapsed:.1f}s"
            )

            for name, result in scene_results.items():
                print(
                    f"    {name}: "
                    f"PSNR={result['PSNR']:.4f}, "
                    f"SSIM={result['SSIM']:.5f}"
                )

    # =====================================
    # 7. FINAL CHECKPOINT
    # =====================================

    torch.save(
        encoder.state_dict(),
        model_dir / "Sen.pth"
    )

    report = {
        "experiment": c.category,
        "epochs": c.num_epoch,
        "training_patches": len(train_dataset),
        "validation_patches": len(val_dataset),
        "validation_scenes": 5,
        "best_validation_psnr": best_psnr,
        "best_validation_ssim": best_ssim,
        "best_psnr_checkpoint": "Sen_Best_PSNR.pth",
        "best_ssim_checkpoint": "Sen_Best_SSIM.pth"
    }

    with open(
        log_dir / "H2C_final_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=4)

    peak = torch.cuda.max_memory_allocated() / 1024**3

    print("\n" + "=" * 65)
    print("H2C CLASSICAL TRAINING COMPLETED")
    print("Best PSNR:", best_psnr)
    print("Best SSIM:", best_ssim)
    print(f"Peak GPU memory: {peak:.2f} GB")
    print("Checkpoints:", model_dir.resolve())
    print("Metrics:", metrics_path.resolve())
    print("=" * 65)


if __name__ == "__main__":
    main()
