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

import essl_analysis_functions

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

def group_spatially_old(df, eps_km=25):
     """ Cluster events spatially using haversine-based DBSCAN. Returns the same DataFrame with a 'cluster_id' column and cluster centers. """ 
     coords = np.radians(df[['LATITUDE', 'LONGITUDE']].values) 
     kms_per_radian = 6371.0088  # Earth radius in km 
     db = DBSCAN(eps=eps_km / kms_per_radian, min_samples=1, metric='haversine') 
     db.fit(coords) 
     df["cluster_id"] = db.labels_ 
     #print how many clusters were found 
     n_clusters = len(set(db.labels_)) 
     print(f" - Found {n_clusters} clusters with eps={eps_km} km") 
     # Compute cluster centers (mean lat/lon) 
     cluster_centers = ( 
         df.groupby("cluster_id")[["LATITUDE", "LONGITUDE"]] 
         .mean() 
         .rename(columns={"LATITUDE": "cluster_lat", "LONGITUDE": "cluster_lon"}) 
     ) 
     df = df.merge(cluster_centers, on="cluster_id", how="left") 
     return df

def group_spatially(df, eps_km=25, start_cluster_id=0):
    """
    Cluster events spatially using haversine-based DBSCAN, with global incremental cluster IDs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'LATITUDE' and 'LONGITUDE' columns.
    eps_km : float, optional
        Clustering radius in kilometers (default = 25 km).
    start_cluster_id : int, optional
        The starting cluster ID to offset the labels, ensuring global uniqueness.
    
    Returns
    -------
    df : pd.DataFrame
        Input DataFrame with added columns:
        - 'cluster_id': unique cluster ID (starting from start_cluster_id)
        - 'cluster_lat', 'cluster_lon': mean cluster coordinates
    n_clusters : int
        Number of clusters found in this batch.
    """
    
    # Handle empty dataframe
    if df.empty:
        print("  - Input dataframe is empty")
        # return empty dataframe with expected columns
        df = df.copy()
        df['cluster_id'] = pd.Series(dtype='int64')
        df['cluster_lat'] = pd.Series(dtype='float64')
        df['cluster_lon'] = pd.Series(dtype='float64')
        return df, 0

    coords = np.radians(df[['LATITUDE', 'LONGITUDE']].values)
    kms_per_radian = 6371.0088  # Earth's radius in km

    db = DBSCAN(eps=eps_km / kms_per_radian, min_samples=1, metric='haversine')
    db.fit(coords)

    # DBSCAN labels (-1 for noise)
    labels = db.labels_

    # Assign raw labels into dataframe first
    df = df.copy()
    df['cluster_id'] = labels

    # Compute unique non-noise labels and how many clusters
    unique_labels = np.unique(labels[labels >= 0])
    n_clusters = len(unique_labels)

    # Offset labels to make them globally unique (map old_label -> new global id)
    label_map = {old: new + start_cluster_id for new, old in enumerate(unique_labels)}
    # Apply mapping: non-negative labels get mapped, -1 stays -1
    df['cluster_id'] = df['cluster_id'].apply(lambda x: label_map[x] if x in label_map else -1)

    # if n_clusters > 0:
    #     print(f"  - Found {n_clusters} clusters (global IDs {start_cluster_id} → {start_cluster_id + n_clusters - 1})")
    # else:
    #     print(f"  - Found {n_clusters} clusters")

    # Compute cluster centers
    cluster_centers = (
        df[df["cluster_id"] >= 0]
        .groupby("cluster_id")[["LATITUDE", "LONGITUDE"]]
        .mean()
        .rename(columns={"LATITUDE": "cluster_lat", "LONGITUDE": "cluster_lon"})
    )
    df = df.merge(cluster_centers, on="cluster_id", how="left")

    return df, n_clusters



def process_events(df, event_type, eps_km=25):
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
    for_cluster_id = 0
    for date, group in df.groupby("date"):
        print(date, len(group))
        clustered, n_clusters = group_spatially(group, eps_km, start_cluster_id=for_cluster_id)
        clustered["day_id"] = str(date)
        all_days.append(clustered)
        for_cluster_id += n_clusters

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


    return df_final, cluster_summary


def plot_clusters(df, title):
    """
    Quick visual check: plot events colored by cluster ID.
    """
    # Handle empty dataframe
    if df is None or df.empty:
        print("plot_clusters: empty dataframe, nothing to plot")
        return

    # Prepare figure with two subplots sharing the same geographic projection
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax_all, ax_centers = axes

    # Determine extent from the full dataset (with a small margin)
    lon_min, lon_max = df["LONGITUDE"].min(), df["LONGITUDE"].max()
    lat_min, lat_max = df["LATITUDE"].min(), df["LATITUDE"].max()
    margin = 0.5
    ax_all.set_extent([lon_min - margin, lon_max + margin, lat_min - margin, lat_max + margin])
    ax_centers.set_extent([lon_min - margin, lon_max + margin, lat_min - margin, lat_max + margin])

    for ax in (ax_all, ax_centers):
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, alpha=0.2)
        ax.add_feature(cfeature.RIVERS, alpha=0.3)

    # Left subplot: all events (uniform blue markers)
    ax_all.scatter(
        df["LONGITUDE"],
        df["LATITUDE"],
        color="blue",
        s=12,
        alpha=0.7,
        transform=ccrs.PlateCarree(),
    )
    ax_all.set_title(f"All events — {title}", fontsize=13, fontweight="bold")

    # Right subplot: cluster centers only
    # Prefer existing cluster center columns, otherwise compute from clustered points
    if "cluster_lat" in df.columns and "cluster_lon" in df.columns:
        centers = df[df["cluster_id"] >= 0][["cluster_id", "cluster_lat", "cluster_lon"]].drop_duplicates(subset=["cluster_id"]) 
        centers = centers.rename(columns={"cluster_lat": "LATITUDE", "cluster_lon": "LONGITUDE"})
    else:
        centers = (
            df[df["cluster_id"] >= 0]
            .groupby("cluster_id")[['LATITUDE', 'LONGITUDE']]
            .mean()
            .reset_index()
            .rename(columns={"LATITUDE": "LATITUDE", "LONGITUDE": "LONGITUDE"})
        )

    if centers is None or centers.empty:
        ax_centers.text(0.5, 0.5, "No cluster centers found", transform=ax_centers.transAxes,
                        ha="center", va="center")
    else:
        ax_centers.scatter(
            centers["LONGITUDE"],
            centers["LATITUDE"],
            color="blue",
            s=70,
            alpha=0.9,
            edgecolor="black",
            transform=ccrs.PlateCarree(),
        )

        ax_centers.set_title(f"Cluster centers — {title}", fontsize=13, fontweight="bold")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, f"{title.replace(' ', '_')}.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)







# ------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------
if __name__ == "__main__":
    print("📂 Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    df = df[['LATITUDE', 'LONGITUDE', 'QC_LEVEL', 'TIME_EVENT', 'TYPE_EVENT', 'PRECIPITATION_AMOUNT', 'MAX_HAIL_DIAMETER']]
    # Exclude events outside April to September
    df["TIME_EVENT"] = pd.to_datetime(df["TIME_EVENT"], utc=True, errors="coerce")
    df = df[df["TIME_EVENT"].dt.month.isin([4, 5, 6, 7, 8, 9])]
   
    
    # Process PRECIP and HAIL separately
    for event_type in ["PRECIP", "HAIL"]:
        if event_type in df["TYPE_EVENT"].unique():
            #print len of df
            print(f"Length of {event_type} DataFrame: {len(df[df['TYPE_EVENT'] == event_type])}")
            grouped_df, summary_df = process_events(df, event_type, EPS_KM)

            if PLOT_CLUSTERS:
                plot_clusters(grouped_df, f"{event_type} clusters (eps={EPS_KM} km)")
                
        print(grouped_df)
        print(summary_df)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        grouped_df.to_csv(f"{OUTPUT_DIR}/{event_type}_grouped.csv", index=False)
        summary_df.to_csv(f"{OUTPUT_DIR}/{event_type}_summary.csv", index=False)

        print(f"✅ Saved grouped {event_type} data to {OUTPUT_DIR}")

    print("🎉 Processing complete!")
