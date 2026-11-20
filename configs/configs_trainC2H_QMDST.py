
import torch
from torchvision import transforms

# ==========================================
# Q-MDST RESEARCH - C2H CONFIGURATION
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# Training parameters
batch_size = 2
num_epoch = 200
lr = 1e-4
decay_epoch = 100
start_epoch = 1
n_threads = 0

# Save stylized database at these epochs
save_list = [150, 160, 170, 180, 190, 200]

# Loss weights: preserve original MDST settings
style_weight = 1.0
content_weight = 1.0
identity_p_weight = 50.0
identity_f_weight = 1.0
identity_pha_weight = 10.0
gan_p_weight = 5.0
gan_f_weight = 0.5

# Input patches are already 256x256
transforms_train = [
    transforms.ToTensor()
]

# Independent experiment name
category = "QMDST_Research_v1"

# Training dataset
content_dir = (
    "../datasets/NH_HAZE_QMDST_v1_train256/"
    "train/clear/"
)

style_dir = (
    "../datasets/NH_HAZE_QMDST_v1_train256/"
    "train/haze/"
)

# Independent output directories
save_stylized_dir = (
    "../results/stylized_database/"
    f"{category}/"
)

save_model_dir = (
    "../results/checkpoints/"
    f"{category}/"
)

save_log_dir = (
    "../results/logs/"
    f"{category}/"
)
