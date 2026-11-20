
from pathlib import Path
from PIL import Image
import csv

ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "datasets" / "NH_HAZE_QMDST_v1"
DEST = ROOT / "datasets" / "NH_HAZE_QMDST_v1_eval256"

PATCH = 256
STRIDE = 256

def get_positions(length):
    if length < PATCH:
        raise ValueError(f"Image dimension {length} is smaller than patch.")

    positions = list(range(0, length - PATCH + 1, STRIDE))
    last = length - PATCH

    if positions[-1] != last:
        positions.append(last)

    return positions


if DEST.exists():
    raise FileExistsError(
        f"Destination already exists: {DEST}. "
        "Do not overwrite existing data."
    )

manifest = []

for split in ["val", "test"]:

    clear_src = SOURCE / split / "clear"
    haze_src = SOURCE / split / "haze"

    clear_dst = DEST / split / "clear"
    haze_dst = DEST / split / "haze"

    clear_dst.mkdir(parents=True, exist_ok=True)
    haze_dst.mkdir(parents=True, exist_ok=True)

    clear_files = sorted(p.name for p in clear_src.iterdir() if p.is_file())
    haze_files = sorted(p.name for p in haze_src.iterdir() if p.is_file())

    if clear_files != haze_files:
        raise ValueError(f"Unmatched pairs in {split}")

    total_patches = 0

    for filename in clear_files:

        with Image.open(clear_src / filename) as im:
            clear = im.convert("RGB")

        with Image.open(haze_src / filename) as im:
            haze = im.convert("RGB")

        if clear.size != haze.size:
            raise ValueError(f"Dimension mismatch: {filename}")

        width, height = clear.size

        xs = get_positions(width)
        ys = get_positions(height)

        stem = Path(filename).stem

        for row, y in enumerate(ys):
            for col, x in enumerate(xs):

                name = f"{stem}_r{row:02d}_c{col:02d}.png"

                box = (x, y, x + PATCH, y + PATCH)

                clear.crop(box).save(clear_dst / name)
                haze.crop(box).save(haze_dst / name)

                manifest.append({
                    "split": split,
                    "source": filename,
                    "patch": name,
                    "x": x,
                    "y": y,
                    "width": PATCH,
                    "height": PATCH
                })

                total_patches += 1

    print(f"{split}: {total_patches} paired patches created.")

manifest_file = DEST / "patch_manifest.csv"

with open(manifest_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split", "source", "patch",
            "x", "y", "width", "height"
        ]
    )
    writer.writeheader()
    writer.writerows(manifest)

print("\nPATCH PREPARATION COMPLETED")
print("Destination:", DEST)
print("Total paired patches:", len(manifest))
print("Manifest:", manifest_file)
