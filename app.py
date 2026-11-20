# -*- coding: utf-8 -*-
"""Q-MDST single-image comparison app. Run: streamlit run app.py"""
import io
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CLASSICAL_CKPT = ROOT / "results/checkpoints/QMDST_Research_v1/H2C_Classical/Sen_Best_PSNR.pth"
HYBRID_CKPT = ROOT / "results/checkpoints/QMDST_Research_v1/QH2C_Hybrid_Full_v1/QSen_Best_PSNR.pth"
DECODER_CKPT = ROOT / "results/checkpoints/QMDST_Research_v1/Tde.pth"
TILE = 256
OVERLAP = 32
QKEY = "quantum_modulation.quantum.weights"

try:
    from models import DSNet
    from models.QDSNet import QDSNetEncoder
    from metrics.metrics import psnr, ssim
    IMPORT_ERROR = None
except Exception as exc:
    DSNet = QDSNetEncoder = psnr = ssim = None
    IMPORT_ERROR = exc

st.set_page_config(page_title="Q-MDST Comparison", page_icon="🌫️", layout="wide")
st.title("Q-MDST | Classical H2C vs Quantum QH2C")
st.caption("Single-image dehazing comparison using the project's trained checkpoints.")

@st.cache_resource(show_spinner=False)
def load_models():
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "Không import được model/dependencies. Hãy chạy trong môi trường "
            f"mdst-dehaze và kiểm tra PennyLane. Chi tiết: {IMPORT_ERROR}"
        )
    paths = [CLASSICAL_CKPT, HYBRID_CKPT, DECODER_CKPT]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Thiếu checkpoint:\n" + "\n".join(missing))

    def load(path):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(obj, dict):
            raise TypeError(f"Checkpoint không phải state_dict: {path}")
        return obj

    c_state, q_state, d_state = map(load, paths)
    if len(c_state) != 124:
        raise RuntimeError(f"Classical checkpoint có {len(c_state)} tensors, kỳ vọng 124.")
    if QKEY not in q_state or q_state[QKEY].numel() != 24:
        raise RuntimeError("Quantum checkpoint không có đúng 24 circuit parameters.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classical = DSNet.Encoder().to(device)
    classical.load_state_dict(c_state, strict=True)
    classical.eval()

    quantum = QDSNetEncoder(base_dim=64, n_qubits=4, n_layers=2).to(device)
    quantum.load_state_dict(q_state, strict=True)
    quantum.eval()

    decoder = DSNet.Decoder().to(device)
    decoder.load_state_dict(d_state, strict=True)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return classical, quantum, decoder, device


def to_tensor(pil):
    arr = np.asarray(ImageOps.exif_transpose(pil).convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().div(255).unsqueeze(0)


def starts(length):
    if length <= TILE:
        return [0]
    stride = TILE - OVERLAP
    values = list(range(0, length - TILE + 1, stride))
    if values[-1] != length - TILE:
        values.append(length - TILE)
    return values


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer(encoder, decoder, tensor, device, progress):
    _, channels, h, w = tensor.shape
    if channels != 3:
        raise ValueError("Ảnh đầu vào phải là RGB.")
    ph, pw = max(h, TILE), max(w, TILE)
    yi = torch.arange(ph).clamp(max=h - 1)
    xi = torch.arange(pw).clamp(max=w - 1)
    padded = tensor.index_select(2, yi).index_select(3, xi)

    coords = [(y, x) for y in starts(ph) for x in starts(pw)]
    y0, x0 = coords[0]
    warm = padded[:, :, y0:y0 + TILE, x0:x0 + TILE].to(device)
    with torch.inference_mode():
        f, s1, s2 = encoder(warm)
        decoder(f, s1, s2)
    sync(device)

    canvas = torch.zeros((3, ph, pw))
    count = torch.zeros((1, ph, pw))
    sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, (y, x) in enumerate(coords, 1):
            tile = padded[:, :, y:y + TILE, x:x + TILE].to(device)
            f, s1, s2 = encoder(tile)
            out = decoder(f, s1, s2)
            if tuple(out.shape[-2:]) != (TILE, TILE):
                raise RuntimeError(f"Decoder output shape unexpected: {tuple(out.shape)}")
            pred = out[0].float().cpu().clamp(0, 1)
            canvas[:, y:y + TILE, x:x + TILE] += pred
            count[:, y:y + TILE, x:x + TILE] += 1
            progress.progress(i / len(coords), text=f"Tile {i}/{len(coords)}")
    sync(device)
    elapsed = time.perf_counter() - t0

    if torch.any(count == 0):
        raise RuntimeError("Có pixel chưa được phủ bởi các tile.")
    result = (canvas / count)[:, :h, :w].clamp(0, 1)
    arr = (result.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr), elapsed, len(coords)


def as_float(pil):
    return np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0


def png(pil):
    b = io.BytesIO()
    pil.save(b, format="PNG")
    return b.getvalue()


with st.sidebar:
    st.subheader("Model configuration")
    st.write("Classical: `DSNet.Encoder`")
    st.write("Quantum: `QDSNetEncoder`")
    st.write("Shared decoder: `Tde.pth` (frozen)")
    st.write("Circuit: 4 qubits · 2 layers · 24 parameters")
    st.divider()
    st.write("Tile: 256 × 256")
    st.write("Overlap: 32 px, predictions averaged")
    st.write("Input: RGB ToTensor [0, 1], no ImageNet normalization")
    st.caption("Inference only. No training or checkpoint writes.")

u1, u2 = st.columns(2)
with u1:
    source_file = st.file_uploader("Upload hazy image", type=["png", "jpg", "jpeg", "webp", "bmp"])
with u2:
    gt_file = st.file_uploader("Optional: upload clear Ground Truth", type=["png", "jpg", "jpeg", "webp", "bmp"])

if source_file is None:
    st.info("Tải ảnh sương mù lên để chạy Classical H2C và Quantum QH2C.")
    st.stop()

try:
    source = ImageOps.exif_transpose(Image.open(io.BytesIO(source_file.getvalue()))).convert("RGB")
except Exception as exc:
    st.error(f"Không đọc được ảnh: {exc}")
    st.stop()

fingerprint = hashlib.sha256(source_file.getvalue()).hexdigest()
if st.session_state.get("input_hash") != fingerprint:
    st.session_state["input_hash"] = fingerprint
    st.session_state.pop("results", None)

reference = None
if gt_file is not None:
    reference = ImageOps.exif_transpose(Image.open(io.BytesIO(gt_file.getvalue()))).convert("RGB")
    if reference.size != source.size:
        st.error("Ground Truth phải có cùng kích thước với ảnh đầu vào.")
        reference = None

a, b = st.columns([1.4, 1])
with a:
    st.image(source, caption="Hazy input", use_container_width=True)
with b:
    st.subheader("Image information")
    st.metric("Resolution", f"{source.width} × {source.height}")
    st.metric("Pixels", f"{source.width * source.height:,}")
    st.caption("Ảnh nhỏ hơn 256 px sẽ được edge-pad, sau đó crop về kích thước gốc.")

if st.button("Run both models", type="primary", use_container_width=True):
    try:
        with st.spinner("Loading checkpoints..."):
            classical, quantum, decoder, device = load_models()
        st.success(f"Models loaded on {device}" + (f" · {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
        tensor = to_tensor(source)

        st.markdown("### Classical H2C")
        pc = st.progress(0)
        out_c, time_c, tiles_c = infer(classical, decoder, tensor, device, pc)

        st.markdown("### Quantum-Hybrid QH2C")
        pq = st.progress(0)
        out_q, time_q, tiles_q = infer(quantum, decoder, tensor, device, pq)

        st.session_state["results"] = {
            "classical": out_c, "quantum": out_q,
            "time_c": time_c, "time_q": time_q,
            "tiles_c": tiles_c, "tiles_q": tiles_q,
        }
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.exception(exc)

results = st.session_state.get("results")
if results:
    out_c, out_q = results["classical"], results["quantum"]
    st.divider()
    st.header("Dehazing outputs")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Classical H2C")
        st.image(out_c, use_container_width=True)
        st.download_button("Download Classical PNG", png(out_c), "QMDST_Classical_H2C.png", "image/png", use_container_width=True)
    with c2:
        st.subheader("Quantum-Hybrid QH2C")
        st.image(out_q, use_container_width=True)
        st.download_button("Download Quantum PNG", png(out_q), "QMDST_Quantum_QH2C.png", "image/png", use_container_width=True)

    st.header("Performance comparison")
    m1, m2, m3 = st.columns(3)
    m1.metric("Classical inference", f"{results['time_c']:.3f} s", f"{results['tiles_c']} tiles")
    m2.metric("Quantum inference", f"{results['time_q']:.3f} s", f"{results['tiles_q']} tiles")
    pct = (results["time_q"] / results["time_c"] - 1) * 100 if results["time_c"] else 0
    m3.metric("Quantum time vs Classical", f"{pct:+.1f}%", "positive = slower", delta_color="inverse")

    if reference is not None:
        try:
            ref = as_float(reference)
            c_psnr, c_ssim = float(psnr(as_float(out_c), ref)), float(ssim(as_float(out_c), ref))
            q_psnr, q_ssim = float(psnr(as_float(out_q), ref)), float(ssim(as_float(out_q), ref))
            st.header("Reference-based quality metrics")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Classical PSNR", f"{c_psnr:.4f} dB")
            k2.metric("Quantum PSNR", f"{q_psnr:.4f} dB", f"{q_psnr-c_psnr:+.4f} dB")
            k3.metric("Classical SSIM", f"{c_ssim:.6f}")
            k4.metric("Quantum SSIM", f"{q_ssim:.6f}", f"{q_ssim-c_ssim:+.6f}")
        except Exception as exc:
            st.error(f"Không tính được PSNR/SSIM: {exc}")
    else:
        st.info("Chưa có Ground Truth: chỉ so sánh hình ảnh và thời gian. PSNR/SSIM không được tính.")

    diff = np.abs(np.asarray(out_c, dtype=np.int16) - np.asarray(out_q, dtype=np.int16)).astype(np.uint8)
    st.subheader("Absolute difference")
    st.image(diff, caption="|Classical output − Quantum output|", use_container_width=True)

    st.caption(
        "Lưu ý phương pháp: giao diện xử lý ảnh đơn bằng tile 256×256, overlap 32 px "
        "và lấy trung bình vùng chồng lấp. Đây là chế độ demo ảnh đơn, không thay thế "
        "benchmark 175 validation patches / 5 scenes trong evaluate_H2C_vs_QH2C.py."
    )
