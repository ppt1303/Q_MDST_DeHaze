
import torch
from torchvision import transforms

# ==========================================
# Q-MDST RESEARCH - C2C TEACHER CONFIG
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# Original C2C training settings
batch_size = 1
num_epoch = 100
lr = 1e-4
decay_epoch = 50
start_epoch = 1
n_threads = 0

# Original loss weights
l1_weight = 1.0
perceptual_weight = 0.1

transforms_train = [
    transforms.ToTensor()
]

category = "QMDST_Research_v1"

# Training: Clear patches
clear_dir = (
    "../datasets/NH_HAZE_QMDST_v1_train256/"
    "train/clear/"
)

# Validation: Clear patches only
# Validation contains 5 original scenes
val_clear_dir = (
    "../datasets/NH_HAZE_QMDST_v1_eval256/"
    "val/clear/"
)

# Independent experiment outputs
save_model_dir = (
    "../results/checkpoints/"
    f"{category}/"
)

save_image_dir = (
    "../results/recimg/"
    f"{category}/C2C/"
)

save_log_dir = (
    "../results/logs/"
    f"{category}/"
)

save_log_path = (
    "../results/logs/"
    f"{category}/train_C2C_console.log"
)
# Validation manifest and original full-resolution images
val_manifest_path = (
    "../datasets/NH_HAZE_QMDST_v1_eval256/"
    "patch_manifest.csv"
)

val_full_clear_dir = (
    "../datasets/NH_HAZE_QMDST_v1/"
    "val/clear/"
)
