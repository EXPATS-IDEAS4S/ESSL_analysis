import os
import glob
import random
import re
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

# ===========================
# CONFIGURATION
# ===========================
paths = {
    "PRECIP": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops/128x128/images/PRECIP/IR_108/png_vmin-vmax_greyscale_CMA",
    "HAIL": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops/128x128/images/HAIL/IR_108/png_vmin-vmax_greyscale_CMA",
    "CONTROL": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops_control/128x128/images/IR_108/png_vmin-vmax_greyscale_CMA"
}

output_gif = "./crops_evolution.gif"
n_examples = 10  # columns
n_rows = len(paths)
duration_per_frame = 0.3  # seconds per frame

# ===========================
# HELPER FUNCTIONS
# ===========================
def extract_timestamp(filename):
    """Extracts datetime string like 20210401T0015 from filename."""
    m = re.search(r"(\d{8}T\d{4})", filename)
    return m.group(1) if m else None


def get_unique_examples(folder, n):
    """Select n random examples (unique prefixes before timestamp)."""
    files = glob.glob(os.path.join(folder, "*.png"))
    prefixes = {}
    for f in files:
        ts = extract_timestamp(os.path.basename(f))
        if ts:
            prefix = os.path.basename(f).replace(f"_{ts}.png", "")
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(f)
    chosen = random.sample(list(prefixes.keys()), min(n, len(prefixes)))
    return {prefix: prefixes[prefix] for prefix in chosen}


def group_by_timestamp(all_examples):
    """Get all available timestamps in chronological order."""
    timestamps = set()
    for ex_dict in all_examples.values():
        for ex_files in ex_dict.values():
            timestamps.update(extract_timestamp(f) for f in ex_files if extract_timestamp(f))
    return sorted(list(timestamps))


# ===========================
# MAIN SCRIPT
# ===========================
print("Selecting examples...")

# Pick examples for each scenario
examples_per_scenario = {}
for scenario, path in paths.items():
    examples_per_scenario[scenario] = get_unique_examples(path, n_examples)

# Collect all timestamps
timestamps = group_by_timestamp(examples_per_scenario)
print(f"Found {len(timestamps)} timestamps.")

frames = []

for ts in tqdm(timestamps, desc="Building GIF frames"):
    fig, axes = plt.subplots(n_rows, n_examples, figsize=(2 * n_examples, 2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    for i, (scenario, ex_dict) in enumerate(examples_per_scenario.items()):
        for j, prefix in enumerate(ex_dict.keys()):
            img_path = os.path.join(paths[scenario], f"{prefix}_{ts}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[i, j].imshow(img, cmap='gray')
            else:
                axes[i, j].imshow(np.zeros((128, 128)), cmap='gray')  # placeholder
            axes[i, j].axis('off')
        axes[i, 0].set_ylabel(scenario, fontsize=12, rotation=90, labelpad=5)
    
    fig.suptitle(f"Timestamp: {ts}", fontsize=14)
    
    # Convert the figure to a numpy array
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(frame)
    plt.close(fig)

# Save as GIF
print(f"Saving GIF to {output_gif} ...")
imageio.mimsave(output_gif, frames, duration=duration_per_frame)
print("✅ GIF saved successfully!")

