import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# CONFIGURATION
# ==============================
datasets = {
    "PRECIP": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops/128x128/nc/PRECIP/1",
    "HAIL": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops/128x128/nc/HAIL/1",
    "CONTROL": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops_control/128x128/nc/1"
}
var_name = "IR_108"
save_dir = "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/figs"
os.makedirs(save_dir, exist_ok=True)


# ==============================
# FUNCTION TO LOAD AND COMPUTE HOURLY STATS
# ==============================
def compute_hourly_stats(folder, var_name):
    files = sorted(glob.glob(os.path.join(folder, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {folder}")

    all_hourly_values = []

    for f in files:
        try:
            ds = xr.open_dataset(f)
            if var_name not in ds:
                continue

            da = ds[var_name]
            if 'time' not in da.dims:
                continue

            # Compute spatial mean per timestamp (collapse lat/lon)
            da_mean = da.mean(dim=[d for d in da.dims if d not in ['time']])
            df = da_mean.to_dataframe().reset_index()
            df['hour'] = pd.to_datetime(df['time']).dt.hour
            all_hourly_values.append(df[['hour', var_name]])
            ds.close()

        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

    df_all = pd.concat(all_hourly_values, ignore_index=True)
    grouped = df_all.groupby("hour")[var_name]

    # Compute several stats
    stats = grouped.agg(['mean', 'std', 'median'])
    stats['p01'] = grouped.quantile(0.01)  # 1st percentile
    return stats, df_all


# ==============================
# COMPUTE STATS FOR EACH DATASET
# ==============================
results = {}
raw_data = {}
for name, path in datasets.items():
    print(f"Processing {name} ...")
    stats, df_all = compute_hourly_stats(path, var_name)
    results[name] = stats
    raw_data[name] = df_all


# ==============================
# PLOT MEAN ± STD
# ==============================
plt.figure(figsize=(10, 6))
colors = {"PRECIP": "tab:blue", "HAIL": "tab:orange", "CONTROL": "tab:green"}

for name, stats in results.items():
    plt.plot(stats.index, stats['mean'], label=f"{name} mean", color=colors[name], linewidth=2)
    plt.fill_between(stats.index,
                     stats['mean'] - stats['std'],
                     stats['mean'] + stats['std'],
                     color=colors[name], alpha=0.2)

plt.xlabel("Hour of day (UTC)")
plt.ylabel("Brightness Temperature (K)")
plt.title("Diurnal Cycle of IR_108 Brightness Temperature (Mean ± Std)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "diurnal_cycle_mean_std.png"), dpi=300)
plt.show()


# ==============================
# PLOT MEDIAN + 1st PERCENTILE
# ==============================
plt.figure(figsize=(10, 6))

for name, stats in results.items():
    plt.plot(stats.index, stats['median'], label=f"{name} median", color=colors[name], linewidth=2)
    plt.plot(stats.index, stats['p01'], linestyle='--', color=colors[name], linewidth=1.5, alpha=0.8)

plt.xlabel("Hour of day (UTC)")
plt.ylabel("Brightness Temperature (K)")
plt.title("Diurnal Cycle of IR_108 BT (Median and 1st Percentile)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "diurnal_cycle_median_p01.png"), dpi=300)
plt.show()


# ==============================
# PLOT BOXES (DISTRIBUTIONS)
# ==============================
plt.figure(figsize=(12, 6))
width = 0.25

for i, (name, df) in enumerate(raw_data.items()):
    df['hour'] = df['hour'].astype(int)
    data_by_hour = [df[df['hour'] == h][var_name].values for h in range(24)]
    plt.boxplot(
        data_by_hour,
        positions=np.arange(24) + i * width,
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor=colors[name], alpha=0.4),
        medianprops=dict(color='black', linewidth=1.5),
        showfliers=False,
        labels=["" for _ in range(24)]
    )

plt.xticks(np.arange(24) + width, range(24))
plt.xlabel("Hour of day (UTC)")
plt.ylabel("Brightness Temperature (K)")
plt.title("Diurnal Distribution of IR_108 Brightness Temperature")
plt.legend(datasets.keys(), loc='best')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "diurnal_cycle_boxplot.png"), dpi=300)
plt.show()