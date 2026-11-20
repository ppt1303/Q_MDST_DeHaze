# -*- coding: utf-8 -*-
"""Inference-only ablation study for Q-MDST.

Compares three models under the same validation protocol:
  A. Classical H2C: original DSNet Encoder + frozen Teacher Decoder.
  B. Full QH2C: QDSNet Encoder + frozen Teacher Decoder.
  C. QH2C block-off: same QH2C checkpoint, but bypass the entire
     quantum_modulation block in forward().

Run from the repository's scripts directory:
    python -u .\\evaluate_Quantum_Ablation.py

No training, optimizer steps, or checkpoint writes are performed.
"""

import sys
import csv
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import configs_trainH2C_QMDST as c
from models import DSNet
from models.QDSNet import QDSNetEncoder
from myutils.dataloader import Dataloader
from metrics.metrics import psnr, ssim

EXPERIMENT = Path("QMDST_Research_v1")
CLASSICAL_CKPT = (
    ROOT / "results/checkpoints" / EXPERIMENT
    / "H2C_Classical/Sen_Best_PSNR.pth"
)
HYBRID_CKPT = (
    ROOT / "results/checkpoints" / EXPERIMENT
    / "QH2C_Hybrid_Full_v1/QSen_Best_PSNR.pth"
)
DECODER_CKPT = ROOT / "results/checkpoints" / EXPERIMENT / "Tde.pth"
MANIFEST = ROOT / "datasets/NH_HAZE_QMDST_v1_eval256/patch_manifest.csv"
FULL_CLEAR_DIR = ROOT / "datasets/NH_HAZE_QMDST_v1/val/clear"

N_PATCHES = 175
N_SCENES = 5
N_CLASSICAL_TENSORS = 124
N_HYBRID_TENSORS = 130
N_QPARAMS = 24
WARMUP_BATCHES = 3


class QH2CBlockOffEncoder(QDSNetEncoder):
    """Same QH2C architecture/state_dict, but bypass QuantumChannelModulation.

    This ablates the complete added quantum_modulation block, including its
    learned pre/post projections and residual scale. It is a module-level
    ablation, not a circuit-parameters-only ablation.
    """

    def forward(self, x):
        # Call the original DSNet encoder implementation directly.
        # This preserves its classical bottleneck and both skip connections,
        # while intentionally skipping self.quantum_modulation(...).
        return DSNet.Encoder.forward(self, x)


def resolve_path(value):
    p = Path(value).expanduser()
    if p.is_absolute():
        return p.resolve()
    for candidate in (Path.cwd() / p, ROOT / "scripts" / p, ROOT / p):
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / "scripts" / p).resolve()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_state(path):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a state_dict: {path}")
    return state


def load_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Validation manifest not found: {MANIFEST}")
    with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    if len(rows) != N_PATCHES:
        raise RuntimeError(
            f"Expected {N_PATCHES} validation patches, got {len(rows)}"
        )
    if len({r["source"] for r in rows}) != N_SCENES:
        raise RuntimeError(f"Expected {N_SCENES} validation scenes")
    if len({r["patch"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate patch names in manifest")
    return rows


def build_loader():
    haze_dir = resolve_path(c.haze_val_dir)
    clear_dir = resolve_path(c.clear_val_dir)
    if not haze_dir.is_dir():
        raise FileNotFoundError(f"Validation haze directory not found: {haze_dir}")
    if not clear_dir.is_dir():
        raise FileNotFoundError(f"Validation clear directory not found: {clear_dir}")

    dataset = Dataloader(
        str(haze_dir),
        str(clear_dir),
        transform=c.transforms_train,
        unaligned=False,
        model="validation",
    )
    if len(dataset) != N_PATCHES:
        raise RuntimeError(f"Expected {N_PATCHES} patches, got {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def forward(encoder, decoder, batch, device):
    haze = batch[0].to(device, non_blocking=(device.type == "cuda"))
    latent, skip1, skip2 = encoder(haze)
    return decoder(latent, skip1, skip2)


def evaluate_model(name, encoder, decoder, loader, manifest, device, image_dir):
    encoder.eval()
    decoder.eval()

    patch_map = {r["patch"]: r for r in manifest}
    canvases = {}
    counts = {}
    seen = set()

    # Same warm-up policy for each model; warm-up is excluded from timing.
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if i >= WARMUP_BATCHES:
                break
            _ = forward(encoder, decoder, batch, device)

    sync(device)
    start = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            patch_name = batch[3][0]
            if patch_name not in patch_map:
                raise RuntimeError(f"Patch absent from manifest: {patch_name}")
            if patch_name in seen:
                raise RuntimeError(f"Duplicate patch from loader: {patch_name}")
            seen.add(patch_name)

            row = patch_map[patch_name]
            output = forward(encoder, decoder, batch, device)
            pred = output[0].float().cpu().clamp(0.0, 1.0)

            source = row["source"]
            x = int(row["x"])
            y = int(row["y"])
            w = int(row["width"])
            h = int(row["height"])

            if source not in canvases:
                target_path = FULL_CLEAR_DIR / source
                if not target_path.is_file():
                    raise FileNotFoundError(
                        f"Full-scene target not found: {target_path}"
                    )
                with Image.open(target_path) as im:
                    full_w, full_h = im.convert("RGB").size
                canvases[source] = torch.zeros(
                    (3, full_h, full_w), dtype=torch.float32
                )
                counts[source] = torch.zeros(
                    (1, full_h, full_w), dtype=torch.float32
                )

            if tuple(pred.shape[-2:]) != (h, w):
                raise RuntimeError(
                    f"Patch size mismatch for {patch_name}: "
                    f"output={tuple(pred.shape[-2:])}, manifest={(h, w)}. "
                    "No silent crop will be applied."
                )

            canvases[source][:, y:y + h, x:x + w] += pred
            counts[source][:, y:y + h, x:x + w] += 1

    sync(device)
    elapsed = time.perf_counter() - start

    if len(seen) != N_PATCHES:
        missing = sorted(set(patch_map) - seen)
        raise RuntimeError(
            f"{name}: evaluated {len(seen)}/{N_PATCHES}; missing={missing[:10]}"
        )

    scene_metrics = {}
    scene_images = {}

    for source in sorted(canvases):
        if torch.any(counts[source] == 0):
            uncovered = int((counts[source] == 0).sum().item())
            raise RuntimeError(f"{name}: {source} has {uncovered} uncovered pixels")

        pred = (canvases[source] / counts[source]).clamp(0.0, 1.0)
        target_path = FULL_CLEAR_DIR / source
        with Image.open(target_path) as im:
            target = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        pred_np = pred.permute(1, 2, 0).numpy()

        scene_metrics[source] = {
            "PSNR": float(psnr(pred_np, target)),
            "SSIM": float(ssim(pred_np, target)),
        }
        scene_images[source] = pred

    if len(scene_metrics) != N_SCENES:
        raise RuntimeError(
            f"{name}: expected {N_SCENES} scenes, got {len(scene_metrics)}"
        )

    image_dir.mkdir(parents=True, exist_ok=True)
    for source, image_tensor in scene_images.items():
        save_image(image_tensor, image_dir / source)

    mean_psnr = float(np.mean([v["PSNR"] for v in scene_metrics.values()]))
    mean_ssim = float(np.mean([v["SSIM"] for v in scene_metrics.values()]))

    return {
        "model": name,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "evaluation_loop_seconds": elapsed,
        "seconds_per_patch": elapsed / N_PATCHES,
        "patches_per_second": N_PATCHES / elapsed,
        "per_scene": scene_metrics,
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    device = torch.device(c.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA but CUDA is unavailable")
    if not FULL_CLEAR_DIR.is_dir():
        raise FileNotFoundError(
            f"Original full-scene clear directory not found: {FULL_CLEAR_DIR}"
        )

    manifest = load_manifest()
    dataset, loader = build_loader()

    # Ensure the loader and manifest refer to exactly the same validation set.
    loader_names = {batch[3][0] for batch in loader}
    manifest_names = {r["patch"] for r in manifest}
    if loader_names != manifest_names:
        raise RuntimeError(
            "DataLoader/manifest mismatch. "
            f"Loader-only={sorted(loader_names - manifest_names)[:10]}, "
            f"Manifest-only={sorted(manifest_names - loader_names)[:10]}"
        )

    classical_state = load_state(CLASSICAL_CKPT)
    hybrid_state = load_state(HYBRID_CKPT)
    decoder_state = load_state(DECODER_CKPT)

    if len(classical_state) != N_CLASSICAL_TENSORS:
        raise RuntimeError(
            f"Expected {N_CLASSICAL_TENSORS} Classical tensors, "
            f"got {len(classical_state)}"
        )
    if len(hybrid_state) != N_HYBRID_TENSORS:
        raise RuntimeError(
            f"Expected {N_HYBRID_TENSORS} Hybrid tensors, "
            f"got {len(hybrid_state)}"
        )

    qkey = "quantum_modulation.quantum.weights"
    if qkey not in hybrid_state or hybrid_state[qkey].numel() != N_QPARAMS:
        raise RuntimeError(
            "Hybrid checkpoint does not contain the expected 24 quantum parameters"
        )

    # Model A: Classical H2C.
    classical = DSNet.Encoder().to(device)
    classical.load_state_dict(classical_state, strict=True)
    classical.eval()

    # Model B: Full QH2C.
    hybrid_full = QDSNetEncoder(
        base_dim=64, n_qubits=4, n_layers=2
    ).to(device)
    hybrid_full.load_state_dict(hybrid_state, strict=True)
    hybrid_full.eval()

    # Model C: same full Hybrid checkpoint and same state_dict, but bypass
    # the entire quantum_modulation block during forward inference.
    hybrid_block_off = QH2CBlockOffEncoder(
        base_dim=64, n_qubits=4, n_layers=2
    ).to(device)
    hybrid_block_off.load_state_dict(hybrid_state, strict=True)
    hybrid_block_off.eval()

    # Confirm both Hybrid variants received exactly the same checkpoint tensors.
    full_sd = hybrid_full.state_dict()
    off_sd = hybrid_block_off.state_dict()
    if list(full_sd.keys()) != list(off_sd.keys()):
        raise RuntimeError("Full and block-off Hybrid architectures have different keys")
    for key in full_sd:
        if not torch.equal(full_sd[key].detach().cpu(), off_sd[key].detach().cpu()):
            raise RuntimeError(f"Hybrid checkpoint mismatch at tensor: {key}")

    # Shared frozen Teacher Decoder for all three models.
    decoder = DSNet.Decoder().to(device)
    decoder.load_state_dict(decoder_state, strict=True)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (
        ROOT / "results/logs" / EXPERIMENT
        / f"Quantum_Ablation_{run_id}"
    )
    image_root = (
        ROOT / "results/dehaze" / EXPERIMENT
        / f"Quantum_Ablation_{run_id}"
    )
    if log_dir.exists() or image_root.exists():
        raise FileExistsError("Ablation output already exists; refusing to overwrite")
    log_dir.mkdir(parents=True, exist_ok=False)
    image_root.mkdir(parents=True, exist_ok=False)

    classical_hash = sha256(CLASSICAL_CKPT)
    hybrid_hash = sha256(HYBRID_CKPT)
    decoder_hash = sha256(DECODER_CKPT)

    print("=" * 76)
    print("Q-MDST - QUANTUM ABLATION STUDY")
    print("=" * 76)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Validation patches:", len(dataset), "| scenes:", N_SCENES)
    print("Classical checkpoint:", CLASSICAL_CKPT)
    print("Hybrid checkpoint:", HYBRID_CKPT)
    print("Shared frozen Teacher Decoder:", DECODER_CKPT)
    print("Classical checkpoint SHA256:", classical_hash)
    print("Hybrid checkpoint SHA256:", hybrid_hash)
    print("Teacher Decoder SHA256:", decoder_hash)
    print("Hybrid quantum circuit parameters:", hybrid_state[qkey].numel())
    print("Full and block-off Hybrid weights: EXACT MATCH")
    print("No training. No optimizer. No checkpoint will be modified.")

    models = [
        (
            "Classical_H2C",
            classical,
            "Original DSNet Encoder checkpoint; no Quantum Modulation.",
        ),
        (
            "Full_Quantum_Hybrid_H2C",
            hybrid_full,
            "QDSNet Encoder with the complete trained Quantum Modulation block.",
        ),
        (
            "QH2C_QuantumBlock_Off",
            hybrid_block_off,
            "Same QH2C checkpoint, but the entire quantum_modulation block is bypassed.",
        ),
    ]

    results = []
    for index, (name, encoder, description) in enumerate(models, start=1):
        print(f"\n[{index}/3] Evaluating {name}...")
        result = evaluate_model(
            name,
            encoder,
            decoder,
            loader,
            manifest,
            device,
            image_root / name,
        )
        result["description"] = description
        results.append(result)
        print(
            f"{name}: PSNR={result['mean_psnr']:.6f} dB | "
            f"SSIM={result['mean_ssim']:.6f} | "
            f"{result['evaluation_loop_seconds']:.2f}s | "
            f"{result['patches_per_second']:.2f} patches/s"
        )

    by_name = {r["model"]: r for r in results}
    classical_result = by_name["Classical_H2C"]
    full_result = by_name["Full_Quantum_Hybrid_H2C"]
    block_off_result = by_name["QH2C_QuantumBlock_Off"]

    summary_rows = []
    for result in results:
        summary_rows.append({
            "model": result["model"],
            "mean_psnr": result["mean_psnr"],
            "mean_ssim": result["mean_ssim"],
            "evaluation_loop_seconds": result["evaluation_loop_seconds"],
            "seconds_per_patch": result["seconds_per_patch"],
            "patches_per_second": result["patches_per_second"],
            "delta_psnr_vs_classical": result["mean_psnr"] - classical_result["mean_psnr"],
            "delta_ssim_vs_classical": result["mean_ssim"] - classical_result["mean_ssim"],
            "delta_psnr_vs_full_hybrid": result["mean_psnr"] - full_result["mean_psnr"],
            "delta_ssim_vs_full_hybrid": result["mean_ssim"] - full_result["mean_ssim"],
        })

    write_csv(
        log_dir / "summary.csv",
        summary_rows,
        [
            "model", "mean_psnr", "mean_ssim", "evaluation_loop_seconds",
            "seconds_per_patch", "patches_per_second",
            "delta_psnr_vs_classical", "delta_ssim_vs_classical",
            "delta_psnr_vs_full_hybrid", "delta_ssim_vs_full_hybrid",
        ],
    )

    scene_rows = []
    scene_names = sorted(classical_result["per_scene"].keys())
    for result in results:
        for scene in scene_names:
            values = result["per_scene"][scene]
            classical_values = classical_result["per_scene"][scene]
            full_values = full_result["per_scene"][scene]
            scene_rows.append({
                "model": result["model"],
                "scene": scene,
                "PSNR": values["PSNR"],
                "SSIM": values["SSIM"],
                "delta_psnr_vs_classical": values["PSNR"] - classical_values["PSNR"],
                "delta_ssim_vs_classical": values["SSIM"] - classical_values["SSIM"],
                "delta_psnr_vs_full_hybrid": values["PSNR"] - full_values["PSNR"],
                "delta_ssim_vs_full_hybrid": values["SSIM"] - full_values["SSIM"],
            })

    write_csv(
        log_dir / "scene_metrics.csv",
        scene_rows,
        [
            "model", "scene", "PSNR", "SSIM",
            "delta_psnr_vs_classical", "delta_ssim_vs_classical",
            "delta_psnr_vs_full_hybrid", "delta_ssim_vs_full_hybrid",
        ],
    )

    report = {
        "experiment": "QMDST_Research_v1/Quantum_Ablation",
        "run_id": run_id,
        "protocol": {
            "validation_patches": N_PATCHES,
            "validation_scenes": N_SCENES,
            "reconstruction": "Average overlapping patch predictions",
            "output_clipping": "[0, 1]",
            "metrics": "metrics.metrics.psnr and metrics.metrics.ssim",
            "shared_decoder": str(DECODER_CKPT),
            "training_performed": False,
            "checkpoints_modified": False,
        },
        "ablation_definition": {
            "type": "whole_quantum_modulation_block_bypass",
            "details": (
                "The block-off model loads the exact same QH2C state_dict as "
                "the full Hybrid model, but its forward method calls the "
                "original DSNet.Encoder.forward directly and skips "
                "quantum_modulation(feat). This removes the entire added "
                "QuantumChannelModulation block, including pre_quantum, "
                "the PennyLane circuit, post_quantum, and residual_scale."
            ),
            "interpretation_limit": (
                "This measures the contribution of the complete added "
                "QuantumChannelModulation block, not the isolated effect "
                "of the 24 variational circuit parameters alone."
            ),
            "full_and_block_off_state_dict_identical": True,
        },
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "pytorch": torch.__version__,
        },
        "checkpoints": {
            "classical": {"path": str(CLASSICAL_CKPT), "sha256": classical_hash},
            "hybrid": {
                "path": str(HYBRID_CKPT),
                "sha256": hybrid_hash,
                "quantum_circuit_parameters": N_QPARAMS,
            },
            "teacher_decoder": {"path": str(DECODER_CKPT), "sha256": decoder_hash},
        },
        "results": {
            result["model"]: result for result in results
        },
        "comparisons": {
            "full_hybrid_minus_classical": {
                "delta_psnr_db": full_result["mean_psnr"] - classical_result["mean_psnr"],
                "delta_ssim": full_result["mean_ssim"] - classical_result["mean_ssim"],
            },
            "block_off_minus_full_hybrid": {
                "delta_psnr_db": block_off_result["mean_psnr"] - full_result["mean_psnr"],
                "delta_ssim": block_off_result["mean_ssim"] - full_result["mean_ssim"],
            },
            "block_off_minus_classical": {
                "delta_psnr_db": block_off_result["mean_psnr"] - classical_result["mean_psnr"],
                "delta_ssim": block_off_result["mean_ssim"] - classical_result["mean_ssim"],
            },
        },
    }

    with open(log_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 76)
    print("QUANTUM ABLATION STUDY COMPLETED")
    print("=" * 76)
    print("Full Hybrid - Classical:",
          f"PSNR={full_result['mean_psnr'] - classical_result['mean_psnr']:+.6f} dB |",
          f"SSIM={full_result['mean_ssim'] - classical_result['mean_ssim']:+.6f}")
    print("Block-off - Full Hybrid:",
          f"PSNR={block_off_result['mean_psnr'] - full_result['mean_psnr']:+.6f} dB |",
          f"SSIM={block_off_result['mean_ssim'] - full_result['mean_ssim']:+.6f}")
    print("Block-off - Classical:",
          f"PSNR={block_off_result['mean_psnr'] - classical_result['mean_psnr']:+.6f} dB |",
          f"SSIM={block_off_result['mean_ssim'] - classical_result['mean_ssim']:+.6f}")
    print("Summary:", log_dir / "summary.csv")
    print("Per-scene metrics:", log_dir / "scene_metrics.csv")
    print("JSON report:", log_dir / "evaluation_report.json")
    print("Reconstructed images:", image_root)
    print("Original checkpoints: PRESERVED")
    print("=" * 76)


if __name__ == "__main__":
    main()
