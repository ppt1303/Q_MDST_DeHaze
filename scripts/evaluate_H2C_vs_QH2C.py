# -*- coding: utf-8 -*-
"""Unified evaluation of Classical H2C vs Quantum-Hybrid QH2C.

Run from the repository's scripts directory:
    python -u .\\evaluate_H2C_vs_QH2C.py

Inference only: no optimizer, no training, and no checkpoint writes.
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
CLASSICAL_CKPT = ROOT / "results/checkpoints" / EXPERIMENT / "H2C_Classical/Sen_Best_PSNR.pth"
HYBRID_CKPT = ROOT / "results/checkpoints" / EXPERIMENT / "QH2C_Hybrid_Full_v1/QSen_Best_PSNR.pth"
DECODER_CKPT = ROOT / "results/checkpoints" / EXPERIMENT / "Tde.pth"
MANIFEST = ROOT / "datasets/NH_HAZE_QMDST_v1_eval256/patch_manifest.csv"
FULL_CLEAR_DIR = ROOT / "datasets/NH_HAZE_QMDST_v1/val/clear"
N_PATCHES, N_SCENES, N_CLASSICAL_TENSORS, N_QPARAMS = 175, 5, 124, 24
WARMUP_BATCHES = 3


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
        raise RuntimeError(f"Expected {N_PATCHES} validation patches, got {len(rows)}")
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
        str(haze_dir), str(clear_dir), transform=c.transforms_train,
        unaligned=False, model="validation",
    )
    if len(dataset) != N_PATCHES:
        raise RuntimeError(f"Expected {N_PATCHES} patches, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        pin_memory=torch.cuda.is_available())
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
    canvases, counts, seen = {}, {}, set()

    # Same short warm-up for both networks; excluded from timed inference.
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
            x, y = int(row["x"]), int(row["y"])
            w, h = int(row["width"]), int(row["height"])
            if source not in canvases:
                target_path = FULL_CLEAR_DIR / source
                if not target_path.is_file():
                    raise FileNotFoundError(f"Full-scene target not found: {target_path}")
                with Image.open(target_path) as im:
                    full_w, full_h = im.convert("RGB").size
                canvases[source] = torch.zeros((3, full_h, full_w), dtype=torch.float32)
                counts[source] = torch.zeros((1, full_h, full_w), dtype=torch.float32)

            if tuple(pred.shape[-2:]) != (h, w):
                raise RuntimeError(
                    f"Patch size mismatch for {patch_name}: output={tuple(pred.shape[-2:])}, "
                    f"manifest={(h, w)}. No silent crop will be applied."
                )
            canvases[source][:, y:y+h, x:x+w] += pred
            counts[source][:, y:y+h, x:x+w] += 1

    sync(device)
    elapsed = time.perf_counter() - start
    if len(seen) != N_PATCHES:
        missing = sorted(set(patch_map) - seen)
        raise RuntimeError(f"{name}: evaluated {len(seen)}/{N_PATCHES}; missing={missing[:10]}")

    scene_metrics, scene_images = {}, {}
    for source in sorted(canvases):
        if torch.any(counts[source] == 0):
            n = int((counts[source] == 0).sum().item())
            raise RuntimeError(f"{name}: {source} has {n} uncovered pixels")
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
        raise RuntimeError(f"{name}: expected {N_SCENES} scenes, got {len(scene_metrics)}")

    image_dir.mkdir(parents=True, exist_ok=True)
    for source, image in scene_images.items():
        save_image(image, image_dir / source)

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


def main():
    device = torch.device(c.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA but CUDA is unavailable")
    if not FULL_CLEAR_DIR.is_dir():
        raise FileNotFoundError(f"Original full-scene clear directory not found: {FULL_CLEAR_DIR}")

    manifest = load_manifest()
    dataset, loader = build_loader()

    # Check loader filenames against manifest before inference.
    loader_names = {batch[3][0] for batch in loader}
    manifest_names = {r["patch"] for r in manifest}
    if loader_names != manifest_names:
        raise RuntimeError(
            "DataLoader/manifest mismatch. "
            f"Loader-only={sorted(loader_names-manifest_names)[:10]}, "
            f"Manifest-only={sorted(manifest_names-loader_names)[:10]}"
        )

    classical_state = load_state(CLASSICAL_CKPT)
    hybrid_state = load_state(HYBRID_CKPT)
    decoder_state = load_state(DECODER_CKPT)
    if len(classical_state) != N_CLASSICAL_TENSORS:
        raise RuntimeError(f"Expected {N_CLASSICAL_TENSORS} Classical tensors, got {len(classical_state)}")
    qkey = "quantum_modulation.quantum.weights"
    if qkey not in hybrid_state or hybrid_state[qkey].numel() != N_QPARAMS:
        raise RuntimeError("Hybrid checkpoint does not contain the expected 24 quantum circuit parameters")

    classical = DSNet.Encoder().to(device)
    classical.load_state_dict(classical_state, strict=True)
    classical.eval()

    hybrid = QDSNetEncoder(base_dim=64, n_qubits=4, n_layers=2).to(device)
    hybrid.load_state_dict(hybrid_state, strict=True)
    hybrid.eval()

    decoder = DSNet.Decoder().to(device)
    decoder.load_state_dict(decoder_state, strict=True)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "results/logs" / EXPERIMENT / f"Unified_Evaluation_{run_id}"
    image_root = ROOT / "results/dehaze" / EXPERIMENT / f"Unified_Evaluation_{run_id}"
    if log_dir.exists() or image_root.exists():
        raise FileExistsError("Evaluation output already exists; refusing to overwrite")
    log_dir.mkdir(parents=True, exist_ok=False)
    image_root.mkdir(parents=True, exist_ok=False)

    print("=" * 72)
    print("Q-MDST - UNIFIED CLASSICAL H2C vs QUANTUM-HYBRID QH2C")
    print("=" * 72)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Validation patches:", len(dataset), "| scenes:", len({r['source'] for r in manifest}))
    print("Shared frozen Teacher Decoder:", DECODER_CKPT)
    print("No training. No checkpoint will be modified.")
    print("Classical checkpoint SHA256:", sha256(CLASSICAL_CKPT))
    print("Hybrid checkpoint SHA256:", sha256(HYBRID_CKPT))
    print("Teacher Decoder SHA256:", sha256(DECODER_CKPT))
    print("Hybrid quantum parameters:", hybrid_state[qkey].numel())

    print("\n[1/2] Evaluating Classical H2C...")
    result_c = evaluate_model("Classical_H2C", classical, decoder, loader, manifest, device,
                              image_root / "Classical_H2C")
    print(f"Classical: PSNR={result_c['mean_psnr']:.6f} dB | SSIM={result_c['mean_ssim']:.6f} | "
          f"{result_c['evaluation_loop_seconds']:.2f}s | {result_c['patches_per_second']:.2f} patches/s")

    print("\n[2/2] Evaluating Quantum-Classical Hybrid H2C...")
    result_q = evaluate_model("Quantum_Hybrid_H2C", hybrid, decoder, loader, manifest, device,
                              image_root / "Quantum_Hybrid_H2C")
    print(f"Hybrid:    PSNR={result_q['mean_psnr']:.6f} dB | SSIM={result_q['mean_ssim']:.6f} | "
          f"{result_q['evaluation_loop_seconds']:.2f}s | {result_q['patches_per_second']:.2f} patches/s")

    delta_psnr = result_q["mean_psnr"] - result_c["mean_psnr"]
    delta_ssim = result_q["mean_ssim"] - result_c["mean_ssim"]
    summary = [
        {k: result_c[k] for k in ("model", "mean_psnr", "mean_ssim", "evaluation_loop_seconds", "seconds_per_patch", "patches_per_second")},
        {k: result_q[k] for k in ("model", "mean_psnr", "mean_ssim", "evaluation_loop_seconds", "seconds_per_patch", "patches_per_second")},
    ]
    with open(log_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader(); writer.writerows(summary)

    scene_rows = []
    for result in (result_c, result_q):
        for scene, values in result["per_scene"].items():
            scene_rows.append({"model": result["model"], "scene": scene,
                               "PSNR": values["PSNR"], "SSIM": values["SSIM"]})
    with open(log_dir / "scene_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "scene", "PSNR", "SSIM"])
        writer.writeheader(); writer.writerows(scene_rows)

    report = {
        "experiment": "QMDST_Research_v1/Unified_Evaluation",
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
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "pytorch": torch.__version__,
        },
        "checkpoints": {
            "classical": {"path": str(CLASSICAL_CKPT), "sha256": sha256(CLASSICAL_CKPT)},
            "hybrid": {"path": str(HYBRID_CKPT), "sha256": sha256(HYBRID_CKPT),
                       "quantum_parameters": N_QPARAMS},
            "teacher_decoder": {"path": str(DECODER_CKPT), "sha256": sha256(DECODER_CKPT)},
        },
        "results": {
            "classical": result_c,
            "quantum_hybrid": result_q,
            "hybrid_minus_classical": {"delta_psnr_db": delta_psnr, "delta_ssim": delta_ssim},
        },
    }
    with open(log_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("UNIFIED EVALUATION COMPLETED")
    print(f"Delta PSNR (Hybrid - Classical): {delta_psnr:+.6f} dB")
    print(f"Delta SSIM (Hybrid - Classical): {delta_ssim:+.6f}")
    print("Summary:", log_dir / "summary.csv")
    print("Per-scene metrics:", log_dir / "scene_metrics.csv")
    print("JSON report:", log_dir / "evaluation_report.json")
    print("Reconstructed images:", image_root)
    print("Original checkpoints: PRESERVED")
    print("=" * 72)


if __name__ == "__main__":
    main()
