from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. Parameters
# =========================

INPUT_IMAGE = "cutout-CDS_P_DSS2_color.jpg"
OUTPUT_DIR = Path("7channel_output")
OUTPUT_DIR.mkdir(exist_ok=True)

SKY_FOV = 11.0        # Full sky image FOV (degrees)
CHANNEL_FOV = 3.5     # Each telescope full FOV (degrees)
SEPARATION = 3.0      # Channel center separation (degrees)

image = Image.open(INPUT_IMAGE).convert("RGB")
W, H = image.size

assert W == H, "This version assumes a square 11° × 11° input image."

pixels_per_degree = W / SKY_FOV
crop_size = round(CHANNEL_FOV * pixels_per_degree)
radius = crop_size / 2

cx, cy = W / 2, H / 2

# =========================
# 2. Channel sky coordinates
# =========================

s = SEPARATION
a = np.sqrt(3) / 2 * s

# x = right, y = up, in degrees
channels = {
    "A": (0, s),
    "B": (a, s / 2),
    "C": (a, -s / 2),
    "D": (0, -s),
    "E": (-a, -s / 2),
    "F": (-a, s / 2),
    "G": (0, 0)
}

# =========================
# 3. Split seven images
# =========================

mosaic = Image.new("RGBA", (W, H), (0, 0, 0, 255))
overview = image.copy()
draw = ImageDraw.Draw(overview)

for name, (dx, dy) in channels.items():

    # Sky angles -> image pixel coordinates
    x = cx + dx * pixels_per_degree
    y = cy - dy * pixels_per_degree

    left = round(x - crop_size / 2)
    top = round(y - crop_size / 2)

    box = (
        left,
        top,
        left + crop_size,
        top + crop_size
    )

    patch = image.crop(box).convert("RGBA")

    # Circular FOV mask
    mask = Image.new("L", (crop_size, crop_size), 0)
    mask_draw = ImageDraw.Draw(mask)

    mask_draw.ellipse(
        (0, 0, crop_size - 1, crop_size - 1),
        fill=255
    )

    patch.putalpha(mask)

    # Save each channel as BMP (no alpha; outside FOV is black)
    channel_bmp = Image.new("RGB", patch.size, (0, 0, 0))
    channel_bmp.paste(patch, mask=patch.getchannel("A"))
    channel_bmp.save(OUTPUT_DIR / f"Channel_{name}.bmp")

    # Reassemble in original sky coordinates
    mosaic.alpha_composite(patch, (left, top))

    # Mark FOV on overview
    r = CHANNEL_FOV / 2 * pixels_per_degree

    color = "orange" if name == "G" else "cyan"

    draw.ellipse(
        (round(x-r), round(y-r),
         round(x+r), round(y+r)),
        outline=color,
        width=max(2, W // 400)
    )

    draw.text(
        (round(x+8), round(y+8)),
        name,
        fill=color
    )

# =========================
# 4. Save results
# =========================

overview.save(OUTPUT_DIR / "FOV_Overview.bmp")
mosaic.convert("RGB").save(OUTPUT_DIR / "Mosaic.bmp")

# =========================
# 5. Visualization
# =========================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image)
axes[0].set_title("Original Sky Image")

axes[1].imshow(overview)
axes[1].set_title("7-Channel FOV Coverage")

axes[2].imshow(mosaic)
axes[2].set_title("Reconstructed Mosaic")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
buf = BytesIO()
plt.savefig(buf, format="png", dpi=200)
buf.seek(0)
Image.open(buf).convert("RGB").save(OUTPUT_DIR / "Comparison.bmp")
plt.show()

print(f"Saved results to: {OUTPUT_DIR.resolve()}")