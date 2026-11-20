
from pathlib import Path
from PIL import Image
import shutil
import csv
import re

# ==========================================
# DATASET CONFIGURATION
# ==========================================

SOURCE = Path(
    r"C:\Users\admin\OneDrive\Documents\Quantum ML\NH-HAZE\NH-HAZE"
)

DEST = Path(
    r"C:\Users\admin\OneDrive\Documents\Quantum ML\MDST-Dehaze\datasets\NH_HAZE_QMDST_v1"
)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# NH-HAZE research split
SPLITS = {
    "train": range(1, 46),
    "val": range(46, 51),
    "test": range(51, 56),
}


# ==========================================
# VALIDATE SOURCE
# ==========================================

if not SOURCE.is_dir():
    raise FileNotFoundError(f"Dataset not found: {SOURCE}")

if DEST.exists():
    raise FileExistsError(
        f"Destination already exists: {DEST}. "
        "Do not overwrite existing experiment data."
    )

def collect_images(pattern):
    result = {}

    for path in SOURCE.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        match = re.fullmatch(pattern, path.stem)

        if match:
            image_id = int(match.group(1))

            if image_id in result:
                raise ValueError(
                    f"Duplicate image ID {image_id}: {path}"
                )

            result[image_id] = path

    return result


gt_images = collect_images(r"(\d+)_GT")
hazy_images = collect_images(r"(\d+)_hazy")

expected_ids = set(range(1, 56))

if set(gt_images) != expected_ids:
    raise ValueError("GT image IDs are incomplete or unexpected.")

if set(hazy_images) != expected_ids:
    raise ValueError("Hazy image IDs are incomplete or unexpected.")

print("Source validation: PASSED")
print("GT images:", len(gt_images))
print("Hazy images:", len(hazy_images))


# ==========================================
# CREATE DATASET COPY
# ==========================================

manifest = []

for split_name, image_ids in SPLITS.items():

    clear_dir = DEST / split_name / "clear"
    haze_dir = DEST / split_name / "haze"

    clear_dir.mkdir(parents=True, exist_ok=True)
    haze_dir.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:

        gt_path = gt_images[image_id]
        haze_path = hazy_images[image_id]

        with Image.open(gt_path) as gt_img:
            gt_size = gt_img.size

        with Image.open(haze_path) as haze_img:
            haze_size = haze_img.size

        if gt_size != haze_size:
            raise ValueError(
                f"Image dimension mismatch: ID {image_id}, "
                f"GT={gt_size}, Hazy={haze_size}"
            )

        filename = f"{image_id:02d}{gt_path.suffix.lower()}"

        clear_output = clear_dir / filename
        haze_output = haze_dir / filename

        shutil.copy2(gt_path, clear_output)
        shutil.copy2(haze_path, haze_output)

        manifest.append({
            "id": image_id,
            "split": split_name,
            "filename": filename,
            "width": gt_size[0],
            "height": gt_size[1],
            "clear_source": str(gt_path),
            "haze_source": str(haze_path),
        })

    print(f"{split_name}: {len(image_ids)} pairs copied.")


# ==========================================
# SAVE MANIFEST
# ==========================================

manifest_path = DEST / "dataset_manifest.csv"

with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "id",
            "split",
            "filename",
            "width",
            "height",
            "clear_source",
            "haze_source",
        ]
    )

    writer.writeheader()
    writer.writerows(manifest)


# ==========================================
# FINAL VALIDATION
# ==========================================

print("\n========== DATASET SUMMARY ==========")

for split_name in SPLITS:

    clear_count = len(
        list((DEST / split_name / "clear").iterdir())
    )

    haze_count = len(
        list((DEST / split_name / "haze").iterdir())
    )

    print(
        f"{split_name}: "
        f"Clear={clear_count}, Hazy={haze_count}"
    )

print("\nManifest:", manifest_path)
print("Dataset created successfully!")
print("Original NH-HAZE files were not modified.")
