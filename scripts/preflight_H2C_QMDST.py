
import os
import sys
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "..")

from models import DSNet
from myutils.dataloader import Dataloader_TrainH2C


ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "datasets"
RESULTS = ROOT / "results"

TRAIN = DATA / "NH_HAZE_QMDST_v1_train256" / "train"

EVAL = DATA / "NH_HAZE_QMDST_v1_eval256" / "val"

STYLE_ROOT = (
    RESULTS / "stylized_database" / "QMDST_Research_v1"
)

CHECKPOINT = RESULTS / "checkpoints" / "QMDST_Research_v1"

VERSIONS = [
    "stylized150",
    "stylized160",
    "stylized170",
    "stylized180",
    "stylized190"
]


def image_files(folder):
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    return {
        p.name for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in
        [".png", ".jpg", ".jpeg"]
    }


def check_dimensions(folder, expected_names):

    for name in expected_names:
        path = folder / name

        with Image.open(path) as im:
            if im.size != (256, 256):
                raise ValueError(
                    f"Invalid image size: {path} = {im.size}"
                )


def main():

    print("=" * 60)
    print("Q-MDST RESEARCH - H2C PREFLIGHT")
    print("=" * 60)

    # ====================================
    # 1. TRAINING FILES
    # ====================================

    clear_dir = TRAIN / "clear"
    clear_names = image_files(clear_dir)

    haze200 = STYLE_ROOT / "stylized200"
    haze_names = image_files(haze200)

    assert len(clear_names) == 180
    assert clear_names == haze_names, (
        "stylized200 filenames do not match Clear!"
    )

    check_dimensions(clear_dir, clear_names)
    check_dimensions(haze200, haze_names)

    print("\n[1] TRAINING PAIRS")
    print("Clear images:", len(clear_names))
    print("Stylized200 images:", len(haze_names))
    print("Filename matching: PASSED")

    # ====================================
    # 2. CONTRASTIVE STYLE DATABASE
    # ====================================

    print("\n[2] STYLE DATABASE")

    for version in VERSIONS:

        folder = STYLE_ROOT / version
        names = image_files(folder)

        assert len(names) == 180, (
            f"{version}: expected 180, got {len(names)}"
        )

        assert names == clear_names, (
            f"Filename mismatch in {version}"
        )

        check_dimensions(folder, names)

        print(version, ":", len(names), "PASSED")

    # ====================================
    # 3. VALIDATION PAIRS
    # ====================================

    print("\n[3] VALIDATION")

    val_clear = EVAL / "clear"
    val_haze = EVAL / "haze"

    clear_val_names = image_files(val_clear)
    haze_val_names = image_files(val_haze)

    assert len(clear_val_names) == 175
    assert clear_val_names == haze_val_names

    manifest_path = EVAL.parent / "patch_manifest.csv"

    with open(
        manifest_path,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r["split"] == "val"]

    assert len(rows) == 175

    scenes = sorted(set(r["source"] for r in rows))

    assert len(scenes) == 5

    print("Validation pairs:", len(clear_val_names))
    print("Original scenes:", len(scenes))
    print("Validation matching: PASSED")

    # ====================================
    # 4. TEACHER CHECKPOINTS
    # ====================================

    print("\n[4] TEACHER CHECKPOINTS")

    Ten_path = CHECKPOINT / "Ten.pth"
    Tde_path = CHECKPOINT / "Tde.pth"

    assert Ten_path.is_file()
    assert Tde_path.is_file()

    Ten = DSNet.Encoder()
    Tde = DSNet.Decoder()

    Ten.load_state_dict(
        torch.load(
            Ten_path,
            map_location="cpu",
            weights_only=True
        ),
        strict=True
    )

    Tde.load_state_dict(
        torch.load(
            Tde_path,
            map_location="cpu",
            weights_only=True
        ),
        strict=True
    )

    Ten.eval()
    Tde.eval()

    print("Ten.pth: STRICT LOAD PASSED")
    print("Tde.pth: STRICT LOAD PASSED")

    # ====================================
    # 5. ORIGINAL H2C DATALOADER
    # ====================================

    print("\n[5] H2C DATALOADER")

    dataset = Dataloader_TrainH2C(
        haze_dir=str(haze200) + os.sep,
        clear_dir=str(clear_dir) + os.sep,
        stylized_database_dir=str(STYLE_ROOT) + os.sep,
        transform=[transforms.ToTensor()]
    )

    assert len(dataset) == 180

    haze, clear, name, candidates, aligned = dataset[0]

    assert haze.shape == (3, 256, 256)
    assert clear.shape == (3, 256, 256)

    assert len(candidates) == 5
    assert len(aligned) == 5

    for item in candidates + aligned:
        assert item.shape == (3, 256, 256)

    print("Dataset length:", len(dataset))
    print("Haze:", tuple(haze.shape))
    print("Clear:", tuple(clear.shape))
    print("Candidate groups:", len(candidates))
    print("Aligned candidate groups:", len(aligned))

    # ====================================
    # 6. TEACHER FORWARD
    # ====================================

    print("\n[6] TEACHER FORWARD")

    with torch.inference_mode():

        feat, x1, x2 = Ten(
            haze.unsqueeze(0)
        )

        output = Tde(feat, x1, x2)

    assert output.shape == (1, 3, 256, 256)

    assert torch.isfinite(output).all()

    print("Output:", tuple(output.shape))
    print("Teacher forward: PASSED")

    # ====================================
    # 7. FINAL
    # ====================================

    print("\n" + "=" * 60)
    print("H2C PREFLIGHT: PASSED")
    print("Teacher checkpoints preserved.")
    print("No training performed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
