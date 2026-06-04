import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter


# ==============================
# CONFIGURATION
# ==============================
DATASETS = {
    "PRECIP": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops/128x128/nc/PRECIP/1",
    "HAIL": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops/128x128/nc/HAIL/1",
    "CONTROL": "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops_control/128x128/nc/1"
}

VAR_NAME = "IR_108"
SAVE_DIR = "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/figs"
os.makedirs(SAVE_DIR, exist_ok=True)

COLD_THRESHOLDS = [220]  # in Kelvin
COLORS = {"PRECIP": "tab:blue", "HAIL": "tab:orange", "CONTROL": "tab:green"}

# ==============================
# HELPER FUNCTIONS
# ==============================
def spatial_smooth(data, size=3):
    """Apply a uniform 3x3 kernel smoothing."""
    return uniform_filter(data, size=size, mode="nearest")


def compute_time_derivative(da):
    """
    Compute temporal derivative (difference) between consecutive timesteps,
    after applying a 3x3 spatial mean filter.
    Output: derivative in K per timestep (assuming regular Δt, e.g., 15 min).
    """
    da_smooth = xr.apply_ufunc(spatial_smooth, da, kwargs={"size": 3})
    da_diff = da_smooth.diff(dim="time")
    # Align time to the midpoint between steps
    da_diff["time"] = da["time"][1:]
    return da_diff


# ==============================
# CORE COMPUTATION
# ==============================
def compute_hourly_stats(folder, var_name):
    """
    Load all nc files, compute:
    - Mean, std, median, p01 (BT)
    - Cold fraction (< thresholds)
    - Spatial std
    - Mean absolute dBT/dt (temporal derivative)
    """
    files = sorted(glob.glob(os.path.join(folder, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {folder}")

    all_records = []

    for f in files:
        try:
            ds = xr.open_dataset(f)
            if var_name not in ds or "time" not in ds[var_name].dims:
                ds.close()
                continue

            da = ds[var_name]

            # --- Compute derivative map ---
            da_dt = compute_time_derivative(da)

            # --- Core stats per timestamp ---
            da_mean = da.mean(dim=["lat", "lon"])
            da_std_spatial = da.std(dim=["lat", "lon"])
            da_dt_mean = da_dt.mean(dim=["lat", "lon"])  # mean dBT/dt

            df = da_mean.to_dataframe().reset_index()
            df["spatial_std"] = da_std_spatial.values
            df["dBTdt"] = np.concatenate([[np.nan], da_dt_mean.values])  # pad first NaN

            # Cold fractions
            for thr in COLD_THRESHOLDS:
                frac = (da < thr).mean(dim=["lat", "lon"])
                df[f"frac_lt_{thr}K"] = frac.values

            df["hour"] = pd.to_datetime(df["time"]).dt.hour
            all_records.append(df)
            ds.close()

        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
            continue

    df_all = pd.concat(all_records, ignore_index=True)
    grouped = df_all.groupby("hour")

    # --- Hourly aggregated stats ---
    stats = grouped[var_name].agg(["mean", "std", "median"])
    stats["p01"] = grouped[var_name].quantile(0.01)
    stats["spatial_std_mean"] = grouped["spatial_std"].mean()
    stats["dBTdt_mean"] = grouped["dBTdt"].mean()
    for thr in COLD_THRESHOLDS:
        stats[f"frac_lt_{thr}K_mean"] = grouped[f"frac_lt_{thr}K"].mean()

    return stats, df_all



# ==============================
# PLOTTING FUNCTIONS
# ==============================
def plot_mean_std(results, save_dir):
    plt.figure(figsize=(10, 6))
    for name, stats in results.items():
        plt.plot(stats.index, stats["mean"], label=f"{name} mean", color=COLORS[name], linewidth=2)
        plt.fill_between(stats.index,
                         stats["mean"] - stats["std"],
                         stats["mean"] + stats["std"],
                         color=COLORS[name], alpha=0.2)
    plt.xlabel("Hour of day (UTC)")
    plt.ylabel("Brightness Temperature (K)")
    plt.title("Diurnal Cycle of IR_108 Brightness Temperature (Mean ± Std)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "diurnal_cycle_mean_std.png"), dpi=300)
    plt.close()


def plot_median_p01(results, save_dir):
    plt.figure(figsize=(10, 6))
    for name, stats in results.items():
        plt.plot(stats.index, stats["median"], label=f"{name} median", color=COLORS[name], linewidth=2)
        plt.plot(stats.index, stats["p01"], linestyle="--", color=COLORS[name], linewidth=1.5, alpha=0.8)
    plt.xlabel("Hour of day (UTC)")
    plt.ylabel("Brightness Temperature (K)")
    plt.title("Diurnal Cycle of IR_108 BT (Median and 1st Percentile)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "diurnal_cycle_median_p01.png"), dpi=300)
    plt.close()


def plot_boxplot(raw_data, save_dir):
    plt.figure(figsize=(12, 6))
    width = 0.25

    for i, (name, df) in enumerate(raw_data.items()):
        data_by_hour = [df[df["hour"] == h][VAR_NAME].values for h in range(24)]
        plt.boxplot(
            data_by_hour,
            positions=np.arange(24) + i * width,
            widths=width,
            patch_artist=True,
            boxprops=dict(facecolor=COLORS[name], alpha=0.4),
            medianprops=dict(color="black", linewidth=1.5),
            showfliers=False,
            labels=["" for _ in range(24)]
        )

    plt.xticks(np.arange(24) + width, range(24))
    plt.xlabel("Hour of day (UTC)")
    plt.ylabel("Brightness Temperature (K)")
    plt.title("Diurnal Distribution of IR_108 Brightness Temperature")
    plt.legend(raw_data.keys(), loc="best")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "diurnal_cycle_boxplot.png"), dpi=300)
    plt.close()


def plot_cold_fraction(results, save_dir):
    plt.figure(figsize=(8, 4))
    for thr in COLD_THRESHOLDS:
        for name, stats in results.items():
            plt.plot(stats.index, stats[f"frac_lt_{thr}K_mean"], label=f"{name} < {thr}K", color=COLORS[name], linewidth=2, alpha=(0.6 if thr == 220 else 1.0))
    plt.xlabel("Hour of day (UTC)", fontsize=14)
    plt.ylabel("Fraction of pixels", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Cold Area Fraction (Brightness Temperature below threshold)", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "diurnal_cycle_cold_fraction.png"), dpi=300)
    plt.close()


def plot_spatial_std(results, save_dir):
    plt.figure(figsize=(8, 4))
    for name, stats in results.items():
        plt.plot(stats.index, stats["spatial_std_mean"], label=f"{name}", color=COLORS[name], linewidth=2)
    plt.xlabel("Hour of day (UTC)", fontsize=14)
    plt.ylabel("Spatial Std (K)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Mean Spatial Standard Deviation of Crops", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "diurnal_cycle_spatial_std.png"), dpi=300)
    plt.close()


def plot_dbt_dt(results, save_dir):
    plt.figure(figsize=(8, 4))
    for name, stats in results.items():
        plt.plot(stats.index, stats["dBTdt_mean"], label=f"{name}", color=COLORS[name], linewidth=2)
    plt.xlabel("Hour of day (UTC)", fontsize=14)
    plt.ylabel("ΔBT/Δt (K per 15 min)", fontsize=14)
    plt.title("Diurnal Cycle of Brightness Temperature Change Rate", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "diurnal_cycle_dBTdt.png"), dpi=300)
    plt.close()

# ==============================
# MAIN PIPELINE
# ==============================
def main():
    results = {}
    raw_data = {}

    for name, path in DATASETS.items():
        print(f"Processing {name} ...")
        stats, df_all = compute_hourly_stats(path, VAR_NAME)
        results[name] = stats
        raw_data[name] = df_all

    # Plot different diagnostics
    #plot_mean_std(results, SAVE_DIR)
    #plot_median_p01(results, SAVE_DIR)
    #plot_boxplot(raw_data, SAVE_DIR)
    #plot_cold_fraction(results, SAVE_DIR)
    #plot_spatial_std(results, SAVE_DIR)
    plot_dbt_dt(results, SAVE_DIR)

    print("✅ All plots saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()
