# -*- coding: utf-8 -*-
"""
Q-MDST-Dehaze: Quantum-Classical Hybrid H2C Training
====================================================

Pilot training script based on train_H2C_QMDST.py.

Important safeguards:
- Loads the Classical H2C student from Sen_Best_PSNR.pth.
- Loads the original frozen Teacher Encoder and Teacher Decoder.
- Adds the PennyLane quantum block through QDSNetEncoder.
- Verifies that all 124 Classical tensors load exactly.
- Verifies that the quantum circuit weights belong to the optimizer.
- Logs quantum gradient norm and quantum weight update magnitude.
- Writes outputs to a separate QH2C_Hybrid_Pilot directory.
- Refuses to overwrite existing files in the Pilot output directories.

Run from the repository's scripts directory:
    python .\\train_QH2C_QMDST.py
"""

import sys
import csv
import json
import time
import random
from pathlib import Path

# Resolve the repository root independently of the current working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import pennylane as qml

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

from configs import configs_trainQH2C_QMDST as c
from models import DSNet
from models.QDSNet import QDSNetEncoder

from myutils.dataloader import (
    Dataloader,
    Dataloader_TrainH2C,
)

from myutils.utils_visual import LambdaLR

from metrics.losses import (
    PerceptualLoss,
    ContrastLoss,
    calc_self_perceptual_loss,
)

from metrics.metrics import psnr, ssim


# ---------------------------------------------------------------------
# EXPERIMENT PATHS
# ---------------------------------------------------------------------

# Deliberately independent from c.category. The supplied configuration
# previously pointed to H2C_Classical, and must never be used as the
# Hybrid output directory.
HYBRID_CATEGORY = Path("QMDST_Research_v1") / "QH2C_Hybrid_Pilot"
CLASSICAL_CATEGORY = (
    Path("QMDST_Research_v1") / "H2C_Classical"
)

CLASSICAL_STUDENT_CHECKPOINT = (
    ROOT
    / "results"
    / "checkpoints"
    / CLASSICAL_CATEGORY
    / "Sen_Best_PSNR.pth"
)

EXPECTED_CLASSICAL_TENSORS = 124
EXPECTED_QUANTUM_CIRCUIT_PARAMETERS = 24
EXPECTED_TRAIN_PATCHES = 180
EXPECTED_VALIDATION_PATCHES = 175
EXPECTED_VALIDATION_SCENES = 5


def resolve_config_path(value):
    """
    Resolve paths stored in the config.

    The original project commonly uses paths relative to scripts/.
    This function supports running from either the repository root
    or the scripts directory.
    """
    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    candidates = [
        Path.cwd() / path,
        ROOT / "scripts" / path,
        ROOT / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Return the most likely original-project interpretation so that
    # the later validation error shows a useful absolute path.
    return (ROOT / "scripts" / path).resolve()


def curriculum_weight(difficulty):
    """
    Keep the original H2C curriculum weighting unchanged.
    """
    diff_list = [18, 20, 25, 27, 32]

    weights = [
        (1 + c.cl_lambda) if difficulty > threshold
        else (1 - c.cl_lambda)
        for threshold in diff_list
    ]

    weights.append(len(diff_list))

    return [value / sum(weights) for value in weights]


def load_manifest():
    """
    Load the original NH-HAZE validation patch manifest.
    """
    manifest_path = (
        ROOT
        / "datasets"
        / "NH_HAZE_QMDST_v1_eval256"
        / "patch_manifest.csv"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Validation manifest not found: {manifest_path}"
        )

    with open(
        manifest_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    rows = [row for row in rows if row["split"] == "val"]

    if len(rows) != EXPECTED_VALIDATION_PATCHES:
        raise RuntimeError(
            "Validation manifest mismatch: "
            f"expected {EXPECTED_VALIDATION_PATCHES}, got {len(rows)}"
        )

    return rows


def get_quantum_weights(encoder):
    """
    Return the trainable PennyLane TorchLayer circuit weights.

    QDSNet.py and the integration smoke test use:
        encoder.quantum_modulation.quantum.weights
    """
    try:
        weights = encoder.quantum_modulation.quantum.weights
    except AttributeError as exc:
        raise RuntimeError(
            "Cannot locate "
            "encoder.quantum_modulation.quantum.weights. "
            "Check models/QDSNet.py and models/QuantumLayer.py."
        ) from exc

    if not isinstance(weights, torch.nn.Parameter):
        raise TypeError(
            "PennyLane circuit weights are not a torch.nn.Parameter."
        )

    return weights


def load_classical_student(encoder, checkpoint_path):
    """
    Load the 124 Classical tensors into QDSNetEncoder.

    Only the new quantum_modulation.* keys may be missing.
    Every original Classical tensor is checked for exact equality.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Classical H2C student checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    classical_state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(classical_state, dict):
        raise TypeError(
            "The Classical checkpoint is not a state_dict."
        )

    if len(classical_state) != EXPECTED_CLASSICAL_TENSORS:
        raise RuntimeError(
            "Unexpected number of Classical tensors: "
            f"expected {EXPECTED_CLASSICAL_TENSORS}, "
            f"got {len(classical_state)}"
        )

    result = encoder.load_state_dict(
        classical_state,
        strict=False,
    )

    expected_missing = {
        key
        for key in encoder.state_dict().keys()
        if key.startswith("quantum_modulation.")
    }

    actual_missing = set(result.missing_keys)
    actual_unexpected = set(result.unexpected_keys)

    if actual_missing != expected_missing:
        raise RuntimeError(
            "Unexpected missing keys while loading Classical weights.\n"
            f"Expected quantum-only missing keys: {sorted(expected_missing)}\n"
            f"Actual missing keys: {sorted(actual_missing)}"
        )

    if actual_unexpected:
        raise RuntimeError(
            "Unexpected keys in Classical checkpoint:\n"
            f"{sorted(actual_unexpected)}"
        )

    # Verify exact preservation of every Classical tensor.
    hybrid_state = encoder.state_dict()

    for key, original_tensor in classical_state.items():
        if key not in hybrid_state:
            raise RuntimeError(
                f"Classical key disappeared from QDSNetEncoder: {key}"
            )

        loaded_tensor = hybrid_state[key].detach().cpu()
        original_tensor = original_tensor.detach().cpu()

        if not torch.equal(loaded_tensor, original_tensor):
            raise RuntimeError(
                f"Classical tensor was not loaded exactly: {key}"
            )

    return len(classical_state), sorted(expected_missing)


def validate(encoder, teacher_decoder, loader, manifest, device):
    """
    Reconstruct the five original validation scenes from 256x256 patches.

    This follows the same validation procedure as the supplied
    Classical H2C training script.
    """
    encoder.eval()
    teacher_decoder.eval()

    patch_map = {
        row["patch"]: row
        for row in manifest
    }

    canvases = {}
    counts = {}

    with torch.inference_mode():
        for batch in loader:
            haze = batch[0].to(device, non_blocking=True)
            patch_name = batch[3][0]

            if patch_name not in patch_map:
                raise RuntimeError(
                    f"Missing validation patch in manifest: {patch_name}"
                )

            row = patch_map[patch_name]

            feat, x1, x2 = encoder(haze)
            output = teacher_decoder(feat, x1, x2)

            prediction = (
                output[0].float().cpu().clamp(0, 1)
            )

            source = row["source"]

            x = int(row["x"])
            y = int(row["y"])
            width = int(row["width"])
            height = int(row["height"])

            if source not in canvases:
                full_path = (
                    ROOT
                    / "datasets"
                    / "NH_HAZE_QMDST_v1"
                    / "val"
                    / "clear"
                    / source
                )

                if not full_path.is_file():
                    raise FileNotFoundError(
                        f"Validation target not found: {full_path}"
                    )

                with Image.open(full_path) as image:
                    target_image = image.convert("RGB")
                    full_width, full_height = target_image.size

                canvases[source] = torch.zeros(
                    (3, full_height, full_width),
                    dtype=torch.float32,
                )

                counts[source] = torch.zeros(
                    (1, full_height, full_width),
                    dtype=torch.float32,
                )

            canvases[source][
                :, y:y + height, x:x + width
            ] += prediction

            counts[source][
                :, y:y + height, x:x + width
            ] += 1

    results = {}
    predictions = {}

    for source in sorted(canvases):
        if torch.any(counts[source] == 0):
            raise RuntimeError(
                f"Validation reconstruction has uncovered pixels: {source}"
            )

        prediction = (
            canvases[source] / counts[source]
        ).clamp(0, 1)

        target_path = (
            ROOT
            / "datasets"
            / "NH_HAZE_QMDST_v1"
            / "val"
            / "clear"
            / source
        )

        with Image.open(target_path) as image:
            target = (
                np.asarray(
                    image.convert("RGB"),
                    dtype=np.float32,
                )
                / 255.0
            )

        prediction_np = (
            prediction.permute(1, 2, 0).numpy()
        )

        scene_psnr = float(psnr(prediction_np, target))
        scene_ssim = float(ssim(prediction_np, target))

        results[source] = {
            "PSNR": scene_psnr,
            "SSIM": scene_ssim,
        }

        predictions[source] = prediction

    if len(results) != EXPECTED_VALIDATION_SCENES:
        raise RuntimeError(
            "Unexpected number of reconstructed validation scenes: "
            f"expected {EXPECTED_VALIDATION_SCENES}, got {len(results)}"
        )

    mean_psnr = float(np.mean([
        result["PSNR"] for result in results.values()
    ]))

    mean_ssim = float(np.mean([
        result["SSIM"] for result in results.values()
    ]))

    encoder.train()
    teacher_decoder.eval()

    return mean_psnr, mean_ssim, results, predictions


def verify_output_directories(model_dir, log_dir, image_dir):
    """
    Refuse to overwrite any previous Pilot artifacts.
    """
    for directory in (model_dir, log_dir, image_dir):
        if directory.exists():
            existing_files = list(directory.rglob("*"))
            existing_files = [
                item for item in existing_files
                if item.is_file()
            ]

            if existing_files:
                preview = "\n".join(
                    str(item) for item in existing_files[:10]
                )
                raise FileExistsError(
                    "Pilot output directory is not empty. "
                    "To protect previous experiments, this script "
                    "will not overwrite it.\n"
                    f"Directory: {directory}\n"
                    f"Existing files:\n{preview}"
                )


def main():
    # -------------------------------------------------------------
    # 0. REPRODUCIBILITY
    # -------------------------------------------------------------
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)

    torch.backends.cudnn.benchmark = True

    device = torch.device(c.device)

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This research script requires CUDA."
        )

    print("=" * 70)
    print("Q-MDST RESEARCH - H2C QUANTUM-CLASSICAL HYBRID TRAINING")
    print("=" * 70)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("PennyLane:", qml.__version__)
    print("Device:", device)

    # -------------------------------------------------------------
    # 1. OUTPUT DIRECTORIES
    # -------------------------------------------------------------
    model_dir = (
        ROOT / "results" / "checkpoints" / HYBRID_CATEGORY
    )
    log_dir = (
        ROOT / "results" / "logs" / HYBRID_CATEGORY
    )
    image_dir = (
        ROOT / "results" / "dehaze" / HYBRID_CATEGORY
    )

    # A hard safety check: Hybrid outputs must not target Classical.
    if HYBRID_CATEGORY == CLASSICAL_CATEGORY:
        raise RuntimeError(
            "Hybrid output directory points to Classical directory."
        )

    verify_output_directories(
        model_dir,
        log_dir,
        image_dir,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = log_dir / "QH2C_metrics.csv"
    report_path = log_dir / "QH2C_final_report.json"

    if metrics_path.exists() or report_path.exists():
        raise FileExistsError(
            "Pilot metrics/report already exist. "
            "No files will be overwritten."
        )

    print("\n[1] OUTPUT")
    print("Hybrid checkpoints:", model_dir)
    print("Hybrid logs:", log_dir)
    print("Hybrid images:", image_dir)
    print("Classical checkpoint directory is read-only.")

    # -------------------------------------------------------------
    # 2. RESOLVE DATA AND TEACHER CHECKPOINT PATHS
    # -------------------------------------------------------------
    haze_dir = resolve_config_path(c.haze_dir)
    clear_dir = resolve_config_path(c.clear_dir)
    stylized_dir = resolve_config_path(c.stylized_dir)

    haze_val_dir = resolve_config_path(c.haze_val_dir)
    clear_val_dir = resolve_config_path(c.clear_val_dir)

    teacher_encoder_path = resolve_config_path(
        c.saved_model_Ten_path
    )
    teacher_decoder_path = resolve_config_path(
        c.saved_model_Tde_path
    )

    required_paths = {
        "Training hazy images": haze_dir,
        "Training clear images": clear_dir,
        "Training stylized images": stylized_dir,
        "Validation hazy patches": haze_val_dir,
        "Validation clear patches": clear_val_dir,
        "Teacher Encoder checkpoint": teacher_encoder_path,
        "Teacher Decoder checkpoint": teacher_decoder_path,
        "Classical student checkpoint": CLASSICAL_STUDENT_CHECKPOINT,
    }

    print("\n[2] PATH VALIDATION")

    for label, path in required_paths.items():
        exists = path.exists()
        print(f"{label}: {exists} | {path}")

        if not exists:
            raise FileNotFoundError(
                f"{label} does not exist:\n{path}"
            )

    # -------------------------------------------------------------
    # 3. TRAIN DATA
    # -------------------------------------------------------------
    train_dataset = Dataloader_TrainH2C(
        str(haze_dir),
        str(clear_dir),
        str(stylized_dir) + "/",
        transform=c.transforms_train,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=c.batch_size,
        shuffle=True,
        num_workers=c.n_threads,
        pin_memory=True,
    )

    if len(train_dataset) != EXPECTED_TRAIN_PATCHES:
        raise RuntimeError(
            "Training dataset mismatch: "
            f"expected {EXPECTED_TRAIN_PATCHES}, "
            f"got {len(train_dataset)}"
        )

    # -------------------------------------------------------------
    # 4. VALIDATION DATA
    # -------------------------------------------------------------
    val_dataset = Dataloader(
        str(haze_val_dir),
        str(clear_val_dir),
        transform=c.transforms_train,
        unaligned=False,
        model="validation",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    manifest = load_manifest()

    if len(val_dataset) != EXPECTED_VALIDATION_PATCHES:
        raise RuntimeError(
            "Validation dataset mismatch: "
            f"expected {EXPECTED_VALIDATION_PATCHES}, "
            f"got {len(val_dataset)}"
        )

    validation_scenes = len(set(
        row["source"] for row in manifest
    ))

    if validation_scenes != EXPECTED_VALIDATION_SCENES:
        raise RuntimeError(
            f"Expected {EXPECTED_VALIDATION_SCENES} validation scenes, "
            f"got {validation_scenes}"
        )

    print("\n[3] DATA")
    print("Training patches:", len(train_dataset))
    print("Validation patches:", len(val_dataset))
    print("Validation scenes:", validation_scenes)
    print("Batch size:", c.batch_size)
    print("Epochs:", c.num_epoch)

    # -------------------------------------------------------------
    # 5. INITIALIZE HYBRID STUDENT AND TEACHERS
    # -------------------------------------------------------------
    encoder = QDSNetEncoder(
        base_dim=64,
        n_qubits=4,
        n_layers=2,
    ).to(device)

    teacher_encoder = DSNet.Encoder().to(device)
    teacher_decoder = DSNet.Decoder().to(device)

    print("\n[4] LOAD CLASSICAL STUDENT")

    classical_tensor_count, quantum_missing_keys = (
        load_classical_student(
            encoder,
            CLASSICAL_STUDENT_CHECKPOINT,
        )
    )

    print("Classical checkpoint:", CLASSICAL_STUDENT_CHECKPOINT)
    print("Classical tensors loaded:", classical_tensor_count)
    print("Classical weights: EXACT MATCH")
    print("New quantum keys:", len(quantum_missing_keys))
    print("Quantum block: NEW INITIALIZATION")

    # -------------------------------------------------------------
    # 6. LOAD AND FREEZE TEACHERS
    # -------------------------------------------------------------
    teacher_encoder.load_state_dict(
        torch.load(
            teacher_encoder_path,
            map_location=device,
            weights_only=True,
        ),
        strict=True,
    )

    teacher_decoder.load_state_dict(
        torch.load(
            teacher_decoder_path,
            map_location=device,
            weights_only=True,
        ),
        strict=True,
    )

    teacher_encoder.eval()
    teacher_decoder.eval()

    for parameter in teacher_encoder.parameters():
        parameter.requires_grad_(False)

    for parameter in teacher_decoder.parameters():
        parameter.requires_grad_(False)

    encoder.train()

    print("\n[5] TEACHERS")
    print("Teacher Encoder: STRICT LOAD PASSED | FROZEN")
    print("Teacher Decoder: STRICT LOAD PASSED | FROZEN")

    # -------------------------------------------------------------
    # 7. QUANTUM PARAMETER VERIFICATION
    # -------------------------------------------------------------
    quantum_weights = get_quantum_weights(encoder)

    if not quantum_weights.requires_grad:
        raise RuntimeError(
            "Quantum circuit weights have requires_grad=False."
        )

    quantum_parameter_count = quantum_weights.numel()

    if quantum_parameter_count != EXPECTED_QUANTUM_CIRCUIT_PARAMETERS:
        raise RuntimeError(
            "Unexpected number of quantum circuit parameters: "
            f"expected {EXPECTED_QUANTUM_CIRCUIT_PARAMETERS}, "
            f"got {quantum_parameter_count}"
        )

    print("\n[6] QUANTUM BLOCK")
    print("Quantum circuit parameters:", quantum_parameter_count)
    print("Quantum weights device:", quantum_weights.device)
    print("Quantum weights trainable:", quantum_weights.requires_grad)

    # -------------------------------------------------------------
    # 8. LOSSES AND OPTIMIZER
    # -------------------------------------------------------------
    loss_l1 = nn.L1Loss().to(device)
    loss_per = PerceptualLoss().to(device)
    loss_cl = ContrastLoss().to(device)

    loss_per.eval()
    loss_cl.eval()

    # All trainable QDSNetEncoder parameters are optimized:
    # Classical student parameters + PennyLane circuit parameters.
    optimizer = optim.Adam(
        encoder.parameters(),
        lr=c.lr,
        betas=(0.5, 0.999),
    )

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    if id(quantum_weights) not in optimizer_parameter_ids:
        raise RuntimeError(
            "Quantum circuit weights are missing from the optimizer."
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=LambdaLR(
            c.num_epoch,
            c.start_epoch,
            c.decay_epoch,
        ).step,
    )

    print("\n[7] OPTIMIZER")
    print("Optimizer: Adam")
    print("Learning rate:", c.lr)
    print("Quantum parameters included: PASSED")
    print("Total trainable parameters:", sum(
        parameter.numel()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    ))

    # -------------------------------------------------------------
    # 9. TRAINING SETUP
    # -------------------------------------------------------------
    best_psnr = 0.0
    best_ssim = 0.0

    columns = [
        "epoch",
        "self_perceptual",
        "l1",
        "perceptual",
        "contrastive",
        "total",
        "quantum_grad_norm",
        "quantum_weight_delta",
        "quantum_grad_nonzero_steps",
        "quantum_update_nonzero_steps",
        "val_psnr_full_scene",
        "val_ssim_full_scene",
        "learning_rate",
        "seconds",
    ]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\n[8] TRAINING START")
    print("Experiment:", HYBRID_CATEGORY.as_posix())
    print("No Classical checkpoint will be modified.")

    # -------------------------------------------------------------
    # 10. FULL TRAINING LOOP
    # -------------------------------------------------------------
    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as metrics_file:

        writer = csv.DictWriter(
            metrics_file,
            fieldnames=columns,
        )
        writer.writeheader()
        metrics_file.flush()

        for epoch in range(1, c.num_epoch + 1):
            epoch_start = time.time()

            encoder.train()
            teacher_encoder.eval()
            teacher_decoder.eval()

            weights = curriculum_weight(best_psnr)

            sums = {
                "self": 0.0,
                "l1": 0.0,
                "per": 0.0,
                "cl": 0.0,
                "total": 0.0,
                "qgrad": 0.0,
                "qdelta": 0.0,
            }

            quantum_grad_nonzero_steps = 0
            quantum_update_nonzero_steps = 0

            learning_rate_used = optimizer.param_groups[0]["lr"]

            for batch_index, batch in enumerate(train_loader, start=1):
                haze = batch[0].to(
                    device,
                    non_blocking=True,
                )

                clear = batch[1].to(
                    device,
                    non_blocking=True,
                )

                candidate_list = batch[3]

                # Frozen Teacher Encoder.
                with torch.no_grad():
                    feat_gt, x1_gt, x2_gt = teacher_encoder(clear)

                # Hybrid Student Encoder.
                feat, x1, x2 = encoder(haze)

                # Frozen Teacher Decoder.
                # Gradients still flow through the decoder operations
                # to the Hybrid Student Encoder.
                output = teacher_decoder(feat, x1, x2)

                loss_self = (
                    calc_self_perceptual_loss(
                        [x1, x2, feat],
                        [x1_gt, x2_gt, feat_gt],
                    )
                    * c.selfperceptual_weight
                )

                loss_l1_value = (
                    loss_l1(output, clear)
                    * c.l1_weight
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
                        weights,
                    )
                    * c.cl1_weight
                )

                total = (
                    loss_self
                    + loss_l1_value
                    + loss_per_value
                    + loss_cl_value
                )

                if not torch.isfinite(total):
                    raise RuntimeError(
                        f"Non-finite loss at epoch {epoch}, "
                        f"batch {batch_index}."
                    )

                optimizer.zero_grad(set_to_none=True)
                total.backward()

                # Measure the quantum circuit gradient before optimizer.step().
                if quantum_weights.grad is None:
                    quantum_grad_norm = 0.0
                else:
                    quantum_grad_norm = float(
                        quantum_weights.grad.detach().norm().item()
                    )

                if not np.isfinite(quantum_grad_norm):
                    raise RuntimeError(
                        "Non-finite quantum gradient at "
                        f"epoch {epoch}, batch {batch_index}."
                    )

                if quantum_grad_norm > 0.0:
                    quantum_grad_nonzero_steps += 1

                # Record circuit weights immediately before the update.
                quantum_weights_before = (
                    quantum_weights.detach().clone()
                )

                optimizer.step()

                # Measure the actual circuit-weight update.
                quantum_weight_delta = float(
                    (
                        quantum_weights.detach()
                        - quantum_weights_before
                    ).norm().item()
                )

                if not np.isfinite(quantum_weight_delta):
                    raise RuntimeError(
                        "Non-finite quantum weight update at "
                        f"epoch {epoch}, batch {batch_index}."
                    )

                if quantum_weight_delta > 0.0:
                    quantum_update_nonzero_steps += 1

                sums["self"] += float(loss_self.item())
                sums["l1"] += float(loss_l1_value.item())
                sums["per"] += float(loss_per_value.item())
                sums["cl"] += float(loss_cl_value.item())
                sums["total"] += float(total.item())
                sums["qgrad"] += quantum_grad_norm
                sums["qdelta"] += quantum_weight_delta

                if batch_index % 20 == 0:
                    print(
                        f"Epoch [{epoch:03d}/{c.num_epoch}] "
                        f"Batch [{batch_index:03d}/{len(train_loader)}] "
                        f"| Loss: {total.item():.5f} "
                        f"| QGrad: {quantum_grad_norm:.8g} "
                        f"| QDelta: {quantum_weight_delta:.8g}"
                    )

            number_of_batches = len(train_loader)

            for key in sums:
                sums[key] /= number_of_batches

            # -----------------------------------------------------
            # 11. VALIDATION
            # -----------------------------------------------------
            (
                val_psnr,
                val_ssim,
                scene_results,
                predictions,
            ) = validate(
                encoder,
                teacher_decoder,
                val_loader,
                manifest,
                device,
            )

            elapsed = time.time() - epoch_start

            # -----------------------------------------------------
            # 12. SAVE LAST HYBRID CHECKPOINT
            # -----------------------------------------------------
            torch.save(
                encoder.state_dict(),
                model_dir / "QSen_last.pth",
            )

            # -----------------------------------------------------
            # 13. SAVE BEST PSNR CHECKPOINT
            # -----------------------------------------------------
            if val_psnr > best_psnr:
                best_psnr = val_psnr

                torch.save(
                    encoder.state_dict(),
                    model_dir / "QSen_Best_PSNR.pth",
                )

                output_dir = image_dir / "val_psnr"
                output_dir.mkdir(parents=True, exist_ok=True)

                for name, prediction in predictions.items():
                    save_image(
                        prediction,
                        output_dir / name,
                    )

            # -----------------------------------------------------
            # 14. SAVE BEST SSIM CHECKPOINT
            # -----------------------------------------------------
            if val_ssim > best_ssim:
                best_ssim = val_ssim

                torch.save(
                    encoder.state_dict(),
                    model_dir / "QSen_Best_SSIM.pth",
                )

                output_dir = image_dir / "val_ssim"
                output_dir.mkdir(parents=True, exist_ok=True)

                for name, prediction in predictions.items():
                    save_image(
                        prediction,
                        output_dir / name,
                    )

            scheduler.step()

            # -----------------------------------------------------
            # 15. SAVE TRAINING STATE
            # -----------------------------------------------------
            torch.save(
                {
                    "epoch": epoch,
                    "encoder": encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "quantum_grad_norm": sums["qgrad"],
                    "quantum_weight_delta": sums["qdelta"],
                    "quantum_grad_nonzero_steps": (
                        quantum_grad_nonzero_steps
                    ),
                    "quantum_update_nonzero_steps": (
                        quantum_update_nonzero_steps
                    ),
                    "classical_initialization": str(
                        CLASSICAL_STUDENT_CHECKPOINT
                    ),
                },
                model_dir / "QH2C_training_state_last.pth",
            )

            # -----------------------------------------------------
            # 16. WRITE METRICS
            # -----------------------------------------------------
            writer.writerow({
                "epoch": epoch,
                "self_perceptual": sums["self"],
                "l1": sums["l1"],
                "perceptual": sums["per"],
                "contrastive": sums["cl"],
                "total": sums["total"],
                "quantum_grad_norm": sums["qgrad"],
                "quantum_weight_delta": sums["qdelta"],
                "quantum_grad_nonzero_steps": (
                    quantum_grad_nonzero_steps
                ),
                "quantum_update_nonzero_steps": (
                    quantum_update_nonzero_steps
                ),
                "val_psnr_full_scene": val_psnr,
                "val_ssim_full_scene": val_ssim,
                "learning_rate": learning_rate_used,
                "seconds": elapsed,
            })

            metrics_file.flush()

            # -----------------------------------------------------
            # 17. EPOCH SUMMARY
            # -----------------------------------------------------
            print("\n" + "-" * 70)
            print(
                f"Epoch [{epoch:03d}/{c.num_epoch}] "
                f"| SelfPer: {sums['self']:.5f} "
                f"| L1: {sums['l1']:.5f} "
                f"| Per: {sums['per']:.5f} "
                f"| CL: {sums['cl']:.5f} "
                f"| Total: {sums['total']:.5f}"
            )

            print(
                f"Validation PSNR: {val_psnr:.5f} "
                f"| Validation SSIM: {val_ssim:.6f}"
            )

            print(
                f"Best PSNR: {best_psnr:.5f} "
                f"| Best SSIM: {best_ssim:.6f}"
            )

            print(
                f"Quantum grad norm (mean): {sums['qgrad']:.10g}"
            )

            print(
                f"Quantum weight delta (mean): {sums['qdelta']:.10g}"
            )

            print(
                "Quantum nonzero gradient steps:",
                quantum_grad_nonzero_steps,
                "/",
                number_of_batches,
            )

            print(
                "Quantum nonzero update steps:",
                quantum_update_nonzero_steps,
                "/",
                number_of_batches,
            )

            print(f"Epoch time: {elapsed:.1f}s")

            for scene_name, result in scene_results.items():
                print(
                    f"    {scene_name}: "
                    f"PSNR={result['PSNR']:.4f}, "
                    f"SSIM={result['SSIM']:.5f}"
                )

            print("-" * 70)

    # -------------------------------------------------------------
    # 18. FINAL HYBRID CHECKPOINT
    # -------------------------------------------------------------
    torch.save(
        encoder.state_dict(),
        model_dir / "QSen.pth",
    )

    # -------------------------------------------------------------
    # 19. FINAL REPORT
    # -------------------------------------------------------------
    report = {
        "experiment": HYBRID_CATEGORY.as_posix(),
        "model": "QDSNetEncoder + frozen DSNet Teacher Decoder",
        "quantum_framework": "PennyLane",
        "pennylane_version": qml.__version__,
        "pytorch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "epochs": c.num_epoch,
        "training_patches": len(train_dataset),
        "validation_patches": len(val_dataset),
        "validation_scenes": validation_scenes,
        "classical_initialization": str(
            CLASSICAL_STUDENT_CHECKPOINT
        ),
        "classical_tensors_loaded_exactly": classical_tensor_count,
        "quantum_circuit_parameters": quantum_parameter_count,
        "teacher_encoder_checkpoint": str(teacher_encoder_path),
        "teacher_decoder_checkpoint": str(teacher_decoder_path),
        "best_validation_psnr": best_psnr,
        "best_validation_ssim": best_ssim,
        "mean_quantum_gradient_norm_last_epoch": sums["qgrad"],
        "mean_quantum_weight_delta_last_epoch": sums["qdelta"],
        "quantum_gradient_nonzero_steps_last_epoch": (
            quantum_grad_nonzero_steps
        ),
        "quantum_update_nonzero_steps_last_epoch": (
            quantum_update_nonzero_steps
        ),
        "checkpoints": {
            "last": "QSen_last.pth",
            "best_psnr": "QSen_Best_PSNR.pth",
            "best_ssim": "QSen_Best_SSIM.pth",
            "final": "QSen.pth",
            "training_state": "QH2C_training_state_last.pth",
        },
        "metrics": "QH2C_metrics.csv",
        "classical_checkpoint_modified": False,
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=4,
            ensure_ascii=False,
        )

    peak_memory_gb = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    print("\n" + "=" * 70)
    print("QH2C HYBRID TRAINING COMPLETED")
    print("=" * 70)
    print("Best validation PSNR:", best_psnr)
    print("Best validation SSIM:", best_ssim)
    print("Mean quantum gradient norm:", sums["qgrad"])
    print("Mean quantum weight delta:", sums["qdelta"])
    print(
        "Quantum nonzero gradient steps:",
        quantum_grad_nonzero_steps,
        "/",
        len(train_loader),
    )
    print(
        "Quantum nonzero update steps:",
        quantum_update_nonzero_steps,
        "/",
        len(train_loader),
    )
    print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")
    print("Hybrid checkpoints:", model_dir.resolve())
    print("Hybrid metrics:", metrics_path.resolve())
    print("Final report:", report_path.resolve())
    print("Classical checkpoints: PRESERVED")
    print("=" * 70)

    if quantum_grad_nonzero_steps == 0:
        print(
            "WARNING: No nonzero quantum gradients were observed. "
            "Do not treat this run as evidence of quantum learning."
        )

    if quantum_update_nonzero_steps == 0:
        print(
            "WARNING: Quantum circuit weights did not change. "
            "Inspect the quantum layer and optimizer."
        )


if __name__ == "__main__":
    main()

