#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Group ESSL event data by type, day, and spatial proximity.

This script:
1. Splits events into PRECIP and HAIL.
2. Groups events occurring on the same day.
3. Uses spatial clustering (DBSCAN with haversine distance) to group close events.
4. Computes cluster centers of mass.
5. Saves per-type grouped data and summaries.
6. Optionally plots clusters on a map.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
INPUT_FILE = "/work/dcorradi/ESSL/conference_analysis_2025/eswd-v2-2021-2025_prec_hail_expats.csv"   # path to your input CSV
OUTPUT_DIR = "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output"
EPS_KM = 150.0                       # clustering distance threshold (km)
PLOT_CLUSTERS = True                # Set to False if you don’t want quick plots

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def group_spatially(df, eps_km=25):
    """
    Cluster events spatially using haversine-based DBSCAN.
    Returns the same DataFrame with a 'cluster_id' column and cluster centers.
    """

    coords = np.radians(df[['LATITUDE', 'LONGITUDE']].values)
    kms_per_radian = 6371.0088  # Earth radius in km

    db = DBSCAN(eps=eps_km / kms_per_radian, min_samples=1, metric='haversine')
    db.fit(coords)
    df["cluster_id"] = db.labels_
    print(df)
    exit()

    # Compute cluster centers (mean lat/lon)
    cluster_centers = (
        df.groupby("cluster_id")[["LATITUDE", "LONGITUDE"]]
          .mean()
          .rename(columns={"LATITUDE": "cluster_lat", "LONGITUDE": "cluster_lon"})
    )
    df = df.merge(cluster_centers, on="cluster_id", how="left")

    return df


def process_events(df, event_type, eps_km=25, output_folder="output"):
    """
    Process events of a given type:
    - group by day
    - cluster spatially
    - save results
    """
    if event_type == 'PRECIP':
        quantity_name = 'PRECIPITATION_AMOUNT'
    elif event_type == 'HAIL':
        quantity_name = 'MAX_HAIL_DIAMETER'
    else:
        raise ValueError("event_type must be 'PRECIP' or 'HAIL'")

    df = df[df["TYPE_EVENT"] == event_type].copy()
    #delete column with quantity name not fitting to type event
    df = df[['LATITUDE', 'LONGITUDE', 'QC_LEVEL', 'TIME_EVENT', 'TYPE_EVENT', quantity_name]]

    df["TIME_EVENT"] = pd.to_datetime(df["TIME_EVENT"], utc=True, errors="coerce")
    df["date"] = df["TIME_EVENT"].dt.date

    all_days = []
    for date, group in df.groupby("date"):
        print(date, len(group))
        clustered = group_spatially(group, eps_km)
        clustered["day_id"] = str(date)
        all_days.append(clustered)

    df_final = pd.concat(all_days, ignore_index=True)

    # Compute cluster summaries
    cluster_summary = (
        df_final.groupby(["day_id", "cluster_id"])
        .agg(
            cluster_lat=("cluster_lat", "first"),
            cluster_lon=("cluster_lon", "first"),
            n_events=("LATITUDE", "count"),
            mean_intensity=(quantity_name, "mean"),
            max_intensity=(quantity_name, "max"),
            start_time=("TIME_EVENT", "min"),
            end_time=("TIME_EVENT", "max")
        )
        .reset_index()
    )

    os.makedirs(output_folder, exist_ok=True)
    df_final.to_csv(f"{output_folder}/{event_type}_grouped.csv", index=False)
    cluster_summary.to_csv(f"{output_folder}/{event_type}_summary.csv", index=False)

    print(f"✅ Saved grouped {event_type} data to {output_folder}")

    return df_final, cluster_summary


def plot_clusters(df, title):
    """
    Quick visual check: plot events colored by cluster ID.
    """
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([df["LONGITUDE"].min()-1, df["LONGITUDE"].max()+1,
                   df["LATITUDE"].min()-1, df["LATITUDE"].max()+1])
    
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, alpha=0.2)
    ax.add_feature(cfeature.RIVERS, alpha=0.3)

    scatter = ax.scatter(df["LONGITUDE"], df["LATITUDE"],
                         c=df["cluster_id"], cmap="tab20", s=20, alpha=0.8,
                         transform=ccrs.PlateCarree())

    plt.title(title, fontsize=14, fontweight="bold")
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------
if __name__ == "__main__":
    print("📂 Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    df = df[['LATITUDE', 'LONGITUDE', 'QC_LEVEL', 'TIME_EVENT', 'TYPE_EVENT', 'PRECIPITATION_AMOUNT', 'MAX_HAIL_DIAMETER']]
    
    # Process PRECIP and HAIL separately
    for event_type in ["PRECIP", "HAIL"]:
        if event_type in df["TYPE_EVENT"].unique():
            grouped_df, summary_df = process_events(df, event_type, EPS_KM, OUTPUT_DIR)

            if PLOT_CLUSTERS:
                plot_clusters(grouped_df, f"{event_type} clusters (eps={EPS_KM} km)")

    print("🎉 Processing complete!")
