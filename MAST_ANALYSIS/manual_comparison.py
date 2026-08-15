import os
import re
from collections import defaultdict
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
DIR_A = "star_plots_2"
DIR_B = "star_plots_1"
OUT_DIR = "combined_output"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# REGEX PATTERNS
# ============================================================
# A: "<starid>_TIC....png"
PATTERN_A = re.compile(r"^(\d+)_.*\.png$", re.IGNORECASE)

# B: "Final_<starid>_Photometry_LS.png"
PATTERN_B = re.compile(r"^Final_(\d+)_.*\.png$", re.IGNORECASE)

# ============================================================
# COLLECT FILES BY STAR ID
# ============================================================
files_A = defaultdict(list)
files_B = defaultdict(list)

for f in os.listdir(DIR_A):
    m = PATTERN_A.match(f)
    if m:
        star_id = m.group(1)
        files_A[star_id].append(os.path.join(DIR_A, f))

for f in os.listdir(DIR_B):
    m = PATTERN_B.match(f)
    if m:
        star_id = m.group(1)
        files_B[star_id].append(os.path.join(DIR_B, f))

# ============================================================
# COMBINE MATCHING IMAGES
# ============================================================
count = 0

for star_id in sorted(set(files_A) & set(files_B)):

    for imgA_path in files_A[star_id]:
        for imgB_path in files_B[star_id]:

            imgA = Image.open(imgA_path).convert("RGB")
            imgB = Image.open(imgB_path).convert("RGB")

            # --- match heights ---
            h = max(imgA.height, imgB.height)

            def pad_height(img, h):
                if img.height == h:
                    return img
                new_img = Image.new("RGB", (img.width, h), (255, 255, 255))
                new_img.paste(img, (0, (h - img.height) // 2))
                return new_img

            imgA = pad_height(imgA, h)
            imgB = pad_height(imgB, h)

            # --- create combined image ---
            combined = Image.new(
                "RGB",
                (imgA.width + imgB.width, h),
                (255, 255, 255)
            )

            combined.paste(imgA, (0, 0))
            combined.paste(imgB, (imgA.width, 0))

            # --- output name ---
            out_name = (
                f"Star_{star_id}_"
                f"A{os.path.basename(imgA_path)}_"
                f"B{os.path.basename(imgB_path)}.png"
            )

            combined.save(os.path.join(OUT_DIR, out_name))
            count += 1

print(f"Done. Created {count} combined images.")
