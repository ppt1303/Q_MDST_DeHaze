
import os
import sys
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
import gradio as gr

from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from configs import configs_inferenceH2C as configs
from models import DSNet

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = ROOT / "results" / "webapp_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_checkpoint(path):
    path = Path(path)

    if path.is_absolute():
        return path

    # Config paths are normally relative to scripts/
    candidate = (ROOT / "scripts" / path).resolve()

    if candidate.exists():
        return candidate

    return (ROOT / path).resolve()


SEN_PATH = resolve_checkpoint(configs.saved_model_Sen_path)
TDE_PATH = resolve_checkpoint(configs.saved_model_Tde_path)


# ==========================================
# 2. LOAD PRETRAINED MODELS
# ==========================================

print("Loading MDST-Dehaze models...")
print("Device:", DEVICE)

if not SEN_PATH.exists():
    raise FileNotFoundError(f"Missing model: {SEN_PATH}")

if not TDE_PATH.exists():
    raise FileNotFoundError(f"Missing model: {TDE_PATH}")

encoder = DSNet.Encoder().to(DEVICE)
decoder = DSNet.Decoder().to(DEVICE)

encoder.load_state_dict(
    torch.load(SEN_PATH, map_location=DEVICE)
)

decoder.load_state_dict(
    torch.load(TDE_PATH, map_location=DEVICE)
)

encoder.eval()
decoder.eval()

print("Models loaded successfully!")


# ==========================================
# 3. IMAGE DEHAZING FUNCTION
# ==========================================

@torch.inference_mode()
def dehaze_image(image, max_side):

    if image is None:
        raise gr.Error("Please upload an image first!")

    original_size = image.size
    image = image.convert("RGB")

    # Resize large images to reduce GPU memory usage
    max_side = int(max_side)

    width, height = image.size

    if max(width, height) > max_side:
        scale = max_side / max(width, height)

        new_size = (
            max(4, round(width * scale)),
            max(4, round(height * scale))
        )

        image = image.resize(
            new_size,
            Image.Resampling.LANCZOS
        )

    width, height = image.size

    # Convert image to tensor
    tensor = to_tensor(image).unsqueeze(0).to(DEVICE)

    # Pad dimensions to multiples of 4
    pad_w = (-width) % 4
    pad_h = (-height) % 4

    if pad_w or pad_h:
        tensor = F.pad(
            tensor,
            (0, pad_w, 0, pad_h),
            mode="replicate"
        )

    # Run MDST-Dehaze
    try:
        feat, x1, x2 = encoder(tensor)
        output = decoder(feat, x1, x2)

    except torch.cuda.OutOfMemoryError:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        raise gr.Error(
            "GPU memory is insufficient. "
            "Please reduce the maximum image size."
        )

    # Remove padding
    output = output[:, :, :height, :width]

    # Convert tensor to PIL image
    output = output[0].detach().cpu().clamp(0, 1)

    result = to_pil_image(output)

    # Restore original image dimensions if resized
    if result.size != original_size:
        result = result.resize(
            original_size,
            Image.Resampling.LANCZOS
        )

    # Save result
    output_path = OUTPUT_DIR / (
        f"{uuid.uuid4().hex}_dehazed.png"
    )

    result.save(output_path, format="PNG")

    print("Dehazing completed:", output_path)

    return result, str(output_path)


# ==========================================
# 4. GRADIO USER INTERFACE
# ==========================================

with gr.Blocks(
    title="MDST-Dehaze | AI Image Restoration"
) as demo:

    gr.Markdown(
        """
        # MDST-Dehaze
        ### AI-Powered Image Dehazing

        Upload your hazy image and let AI restore
        its clarity and visibility.
        """
    )

    with gr.Row():

        input_image = gr.Image(
            type="pil",
            label="INPUT IMAGE | HAZY",
            height=420
        )

        output_image = gr.Image(
            type="pil",
            label="OUTPUT IMAGE | DEHAZED",
            height=420
        )

    max_side = gr.Slider(
        minimum=512,
        maximum=2048,
        value=1536,
        step=128,
        label="Maximum processing image size (pixels)"
    )

    dehaze_button = gr.Button(
        "DEHAZE",
        variant="primary"
    )

    download_file = gr.File(
        label="DOWNLOAD DEHAZED IMAGE"
    )

    dehaze_button.click(
        fn=dehaze_image,
        inputs=[input_image, max_side],
        outputs=[output_image, download_file]
    )

    gr.Markdown(
        """
        ---
        **Powered by MDST-Dehaze | PyTorch | NVIDIA CUDA**

        GPU acceleration enabled when available.
        """
    )


# ==========================================
# 5. LAUNCH APPLICATION
# ==========================================

if __name__ == "__main__":

    demo.queue().launch(
        inbrowser=True,
        server_name="127.0.0.1"
    )
