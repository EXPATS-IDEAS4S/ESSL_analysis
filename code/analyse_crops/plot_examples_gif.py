import os
import glob
import random
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

# ===================================
# CONFIGURATION
# ===================================
nc_paths = {
    "PRECIP": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops/128x128/nc/PRECIP/1",
    "HAIL": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops/128x128/nc/HAIL/1",
    "CONTROL": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops_control/128x128/nc/1",
}

img_paths = {
    "PRECIP": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops/128x128/images/PRECIP/IR_108/png_vmin-vmax_greyscale_CMA",
    "HAIL": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops/128x128/images/HAIL/IR_108/png_vmin-vmax_greyscale_CMA",
    "CONTROL": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops_control/128x128/images/IR_108/png_vmin-vmax_greyscale_CMA",
}

output_gif = "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/figs/crops_evolution_from_nc.gif"
n_examples = 10        # columns
duration_per_frame = 0.25  # seconds per frame
expected_per_day = 96      # 15-min intervals * 24h

# ===================================
# HELPER FUNCTIONS
# ===================================
def extract_date_from_nc(filename):
    """Extract date (yyyy_mm_dd) from NetCDF filename."""
    base = os.path.basename(filename)
    return base.split("_")[0]

def select_random_nc_dates(nc_folder, n):
    """Pick n random NetCDF files and extract their dates."""
    nc_files = sorted(glob.glob(os.path.join(nc_folder, "*.nc")))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {nc_folder}")
    chosen = random.sample(nc_files, min(n, len(nc_files)))
    return [extract_date_from_nc(f) for f in chosen]

def find_day_images(img_folder, date):
    """Find the first 96 images for a given date (full day evolution)."""
    imgs = sorted(glob.glob(os.path.join(img_folder, f"{date}*.png")))
    if len(imgs) < expected_per_day:
        raise ValueError(f"⚠️ Found only {len(imgs)} images for {date} in {img_folder}, expected ≥ {expected_per_day}")
    return imgs[:expected_per_day]

# ===================================
# MAIN
# ===================================
print("Selecting random examples (days) for each scenario...")

day_images = {}
timestamps = [f"{str(h).zfill(2)}{str(m).zfill(2)}"
              for h in range(24)
              for m in (0, 15, 30, 45)]  # 15-min intervals → 96 frames

for scenario, nc_dir in nc_paths.items():
    dates = select_random_nc_dates(nc_dir, n_examples)
    print(f" → {scenario}: selected {len(dates)} days → {dates}")
    imgs_per_day = []
    for d in dates:
        imgs = find_day_images(img_paths[scenario], d)
        imgs_per_day.append(imgs)
    day_images[scenario] = imgs_per_day

# ===================================
# BUILD ANIMATION
# ===================================
print("Building animation grid...")

n_rows = len(nc_paths)
fig, axes = plt.subplots(
    n_rows, n_examples, 
    figsize=(1.8 * n_examples, 1.8 * n_rows),
    constrained_layout=False
)
plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.90, bottom=0.05)

if n_rows == 1:
    axes = np.expand_dims(axes, axis=0)

# Initialize placeholders (fix crop size, consistent scaling)
ims = [[axes[i, j].imshow(np.zeros((128, 128)), cmap="gray", vmin=0, vmax=255, aspect='equal')
        for j in range(n_examples)] for i in range(n_rows)]

# Add row and column labels
for i, scenario in enumerate(nc_paths.keys()):
    for j in range(n_examples):
        axes[i, j].axis("off")

    # Label scenario on the left side
    # shift text a bit to the right closer to the images
    fig.text(0.05, 1 - (i + 0.5) / n_rows, scenario, 
             va='center', ha='right', fontsize=12, weight='bold')
    
# Add column titles (“Random 1”, “Random 2”, …)
for j in range(n_examples):
    axes[0, j].set_title(f"Random {j + 1}", fontsize=10, pad=4)

frames = []

# Loop through 96 timestamps
for frame_idx in tqdm(range(expected_per_day), desc="Creating frames"):
    for i, scenario in enumerate(day_images.keys()):
        for j, imgs_day in enumerate(day_images[scenario]):
            if frame_idx < len(imgs_day):
                img_path = imgs_day[frame_idx]
                img = Image.open(img_path)
                ims[i][j].set_data(img)
            else:
                ims[i][j].set_data(np.zeros((128, 128)))
    fig.suptitle(f"Timestamp {frame_idx + 1:02d}/{expected_per_day}  ({timestamps[frame_idx]})", fontsize=14)

    # Render to array (ensures consistent crop sizing)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame.copy())

plt.close(fig)

# ===================================
# SAVE GIF
# ===================================
os.makedirs(os.path.dirname(output_gif), exist_ok=True)
print(f"Saving GIF to {output_gif} ...")
imageio.mimsave(output_gif, frames, duration=duration_per_frame)
print("✅ GIF saved successfully!")
