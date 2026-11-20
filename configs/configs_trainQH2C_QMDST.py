
import torch
from torchvision import transforms

# ==========================================
# Q-MDST RESEARCH - H2C QUANTUM HYBRID PILOT
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# Original MDST H2C settings
batch_size = 1
num_epoch = 1
lr = 1e-4
decay_epoch = 0
start_epoch = 0
n_threads = 0

# Original loss weights
cl_lambda = 0.25

l1_weight = 10.0
selfperceptual_weight = 1.0
perceptual_weight = 1.0
cl1_weight = 0.1
cl2_weight = 0.1

transforms_train = [
    transforms.ToTensor()
]

unsupervised = False

# ==========================================
# DATASET
# ==========================================

category = "QMDST_Research_v1/QH2C_Hybrid_Pilot"

clear_dir = (
    "../datasets/NH_HAZE_QMDST_v1_train256/"
    "train/clear/"
)

haze_dir = (
    "../results/stylized_database/"
    "QMDST_Research_v1/stylized200/"
)

stylized_dir = (
    "../results/stylized_database/"
    "QMDST_Research_v1/"
)

# ==========================================
# VALIDATION
# ==========================================

clear_val_dir = (
    "../datasets/NH_HAZE_QMDST_v1_eval256/"
    "val/clear/"
)

haze_val_dir = (
    "../datasets/NH_HAZE_QMDST_v1_eval256/"
    "val/haze/"
)

# ==========================================
# PRETRAINED TEACHER
# ==========================================

saved_model_Ten_path = (
    "../results/checkpoints/"
    "QMDST_Research_v1/Ten.pth"
)

saved_model_Tde_path = (
    "../results/checkpoints/"
    "QMDST_Research_v1/Tde.pth"
)

# ==========================================
# INDEPENDENT OUTPUT
# ==========================================

save_image_dir = (
    "../results/dehaze/"
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

# ==========================================
# PRETRAINED CLASSICAL STUDENT
# ==========================================

saved_model_student_path = (
    "../results/checkpoints/"
    "QMDST_Research_v1/"
    "H2C_Classical/"
    "Sen_Best_PSNR.pth"
)