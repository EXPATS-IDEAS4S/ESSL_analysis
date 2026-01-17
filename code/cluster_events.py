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
from sklearn.cluster import DBSCAN
import os

import essl_analysis_functions, plot_utils, data_utils
from data_utils import add_msg_time_slot, group_spatially, filter_by_event_years
from plot_utils import plot_clusters, plot_random_days_msg_clusters

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
INPUT_FILE = "/work/dcorradi/ESSL/1st_paper/eswd-v2-2012-2025_expats.csv"   # path to your input CSV             
PLOT_CLUSTERS = True                # Set to False if you don’t want quick plots
YEARS = [2014, 2015, 2016, 2018, 2019, 2020, 2022, 2023, 2024]  # Years to process
MONTHS = [4, 5, 6, 7, 8, 9]          # Months to include (April to September)
EVENT_TYPES = ['PRECIP', 'HAIL']  # Event types to process
EPS_KM = 150  # clustering distance threshold (km)
FREQ = "15min"                     # Time slot frequenc
OUTPUT_DIR = f"/work/dcorradi/ESSL/1st_paper/grouped_output_{FREQ}_eps{EPS_KM}km"  # output directory
# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def process_all_events(df, eps_km=100, freq="15min"):
    """
    Process all events together:
    - group by MSG time slot
    - spatial DBSCAN
    - diagnose cluster type (PRECIP / HAIL / MIXED)
    """

    df = df.copy()
    df["TIME_EVENT"] = pd.to_datetime(df["TIME_EVENT"], utc=True, errors="coerce")

    # Assign MSG-aligned time slots
    df = add_msg_time_slot(df, time_col="TIME_EVENT", freq=freq)

    all_slots = []
    global_cluster_id = 0

    for slot_time, group in df.groupby("time_slot"):
        print(slot_time, len(group))

        clustered, n_clusters = group_spatially(
            group,
            eps_km=eps_km,
            start_cluster_id=global_cluster_id
        )

        clustered["time_slot"] = slot_time
        clustered["day_id"] = slot_time.date().isoformat()

        all_slots.append(clustered)
        global_cluster_id += n_clusters

    df_final = pd.concat(all_slots, ignore_index=True)

    # --------------------------------------------------
    # NEW: diagnose cluster event type
    # --------------------------------------------------
    cluster_type = (
        df_final
        .groupby(["time_slot", "cluster_id"])["TYPE_EVENT"]
        .apply(lambda x: "MIXED" if x.nunique() > 1 else x.iloc[0])
        .rename("cluster_event_type")
        .reset_index()
    )

    df_final = df_final.merge(
        cluster_type,
        on=["time_slot", "cluster_id"],
        how="left"
    )

    # --------------------------------------------------
    # Cluster summary
    # --------------------------------------------------
    cluster_summary = (
        df_final.groupby(["time_slot", "cluster_id"])
        .agg(
            cluster_event_type=("cluster_event_type", "first"),
            cluster_lat=("cluster_lat", "first"),
            cluster_lon=("cluster_lon", "first"),
            n_events=("LATITUDE", "count"),
            n_precip=("TYPE_EVENT", lambda x: (x == "PRECIP").sum()),
            n_hail=("TYPE_EVENT", lambda x: (x == "HAIL").sum()),
            start_time=("TIME_EVENT", "min"),
            end_time=("TIME_EVENT", "max"),
        )
        .reset_index()
    )

    return df_final, cluster_summary





# ------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------
if __name__ == "__main__":
    print("📂 Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    df = df[['LATITUDE', 'LONGITUDE', 'QC_LEVEL', 'TIME_EVENT', 'TYPE_EVENT', 'PRECIPITATION_AMOUNT', 'MAX_HAIL_DIAMETER']]
    
    #take only the years in YEARS list
    print(f"📅 Filtering events for years: {YEARS}...")
    df = filter_by_event_years(df, YEARS)

    # Exclude events outside April to September
    df["TIME_EVENT"] = pd.to_datetime(df["TIME_EVENT"], utc=True, errors="coerce")
    df = df[df["TIME_EVENT"].dt.month.isin(MONTHS)]
   
    
    print("🌀 Processing all events together...")
    grouped_df, summary_df = process_all_events(df, EPS_KM, FREQ)

    if PLOT_CLUSTERS:
        plot_clusters(grouped_df, f"All clusters (eps={EPS_KM} km)", OUTPUT_DIR)
        plot_random_days_msg_clusters(
            grouped_df,
            n_days=20,
            max_slots=10,
            output_dir=OUTPUT_DIR,
            seed=12
        )
          
    print(grouped_df)
    print(summary_df)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grouped_df.to_csv(f"{OUTPUT_DIR}/events_grouped.csv", index=False)
    summary_df.to_csv(f"{OUTPUT_DIR}/events_summary.csv", index=False)
    print(f"✅ Saved grouped data to {OUTPUT_DIR}")

    print("🎉 Processing complete!")
