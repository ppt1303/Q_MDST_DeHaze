
from pathlib import Path
from PIL import Image
import random
import csv

ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "datasets" / "NH_HAZE_QMDST_v1" / "train"

DEST = ROOT / "datasets" / "NH_HAZE_QMDST_v1_train256"

PATCH = 256
SEED = 2026

# Do not overwrite existing experiment data
if DEST.exists():
    raise FileExistsError(
        f"Destination already exists: {DEST}"
    )

clear_dir = SOURCE / "clear"
haze_dir = SOURCE / "haze"

def collect_images(folder):
    result = {}

    for p in folder.iterdir():
        if p.is_file() and p.stem.isdigit():
            image_id = int(p.stem)

            if image_id in result:
                raise ValueError(f"Duplicate ID: {image_id}")

            result[image_id] = p

    return result


clear_images = collect_images(clear_dir)
haze_images = collect_images(haze_dir)

expected = set(range(1, 46))

if set(clear_images) != expected:
    raise ValueError("Clear training IDs are incomplete.")

if set(haze_images) != expected:
    raise ValueError("Hazy training IDs are incomplete.")

# Validate all image pairs before creating output
for image_id in sorted(expected):
    with Image.open(clear_images[image_id]) as a:
        clear_size = a.size

    with Image.open(haze_images[image_id]) as b:
        haze_size = b.size

    if clear_size != haze_size:
        raise ValueError(f"Size mismatch: {image_id}")

    if clear_size[0] < 512 or clear_size[1] < 512:
        raise ValueError(f"Image too small: {image_id}")

print("Source validation: PASSED")

clear_out = DEST / "train" / "clear"
haze_out = DEST / "train" / "haze"

clear_out.mkdir(parents=True)
haze_out.mkdir(parents=True)

manifest = []

for image_id in sorted(expected):

    with Image.open(clear_images[image_id]) as im:
        clear = im.convert("RGB")

    with Image.open(haze_images[image_id]) as im:
        haze = im.convert("RGB")

    width, height = clear.size

    x_ranges = [
        (0, width // 2 - PATCH),
        (width // 2, width - PATCH)
    ]

    y_ranges = [
        (0, height // 2 - PATCH),
        (height // 2, height - PATCH)
    ]

    rng = random.Random(SEED + image_id)

    positions = [
        (x_ranges[0], y_ranges[0]),
        (x_ranges[1], y_ranges[0]),
        (x_ranges[0], y_ranges[1]),
        (x_ranges[1], y_ranges[1])
    ]

    for patch_id, (xr, yr) in enumerate(positions):

        x = rng.randint(xr[0], xr[1])
        y = rng.randint(yr[0], yr[1])

        box = (
            x, y,
            x + PATCH,
            y + PATCH
        )

        filename = f"{image_id:02d}_q{patch_id:02d}.png"

        clear.crop(box).save(clear_out / filename)
        haze.crop(box).save(haze_out / filename)

        manifest.append({
            "image_id": image_id,
            "patch_id": patch_id,
            "filename": filename,
            "x": x,
            "y": y,
            "width": PATCH,
            "height": PATCH
        })

manifest_path = DEST / "train_patch_manifest.csv"

with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_id", "patch_id", "filename",
            "x", "y", "width", "height"
        ]
    )

    writer.writeheader()
    writer.writerows(manifest)

clear_count = len(list(clear_out.glob("*.png")))
haze_count = len(list(haze_out.glob("*.png")))

assert clear_count == 180
assert haze_count == 180

print("\n========== TRAIN PATCH SUMMARY ==========")
print("Original scenes:", 45)
print("Patches per scene:", 4)
print("Clear patches:", clear_count)
print("Hazy patches:", haze_count)
print("Patch size:", PATCH, "x", PATCH)
print("Manifest:", manifest_path)
print("Training patches created successfully!")
