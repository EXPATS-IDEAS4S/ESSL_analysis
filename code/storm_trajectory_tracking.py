#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Storm trajectory tracking from 15-min clustered ESSL events.

Features:
- Global storm IDs
- Temporal tolerance up to 60 min
- Distance-based matching (eps-like threshold)
- Gap interpolation
- ±2 h lifecycle extrapolation
- Event-type consistency (PRECIP / HAIL / MIXED)
- No merging / splitting
"""

import pandas as pd
import numpy as np
from haversine import haversine, Unit
from datetime import timedelta
import os

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
EPS_KM = 150  # clustering distance threshold (km)
FREQ = "15min"                     # Time slot frequenc
EVENT_TYPES = ['PRECIP', 'HAIL']  # Event types to process
GROUPED_DIR = f"/work/dcorradi/ESSL/1st_paper/grouped_output_{FREQ}_eps{EPS_KM}km"  # output directory
OUTPUT_DIR = f"{GROUPED_DIR}/trajectories"

TIME_COL = "time_slot"
LAT_COL = "cluster_lat"
LON_COL = "cluster_lon"
TYPE_COL = "cluster_event_type"

TIME_STEP_MIN = 15
MAX_BACKWARD_MIN = 60
MAX_DIST_KM = 200.0
MAX_SPEED_KMH = 150.0 

EXTEND_MIN = 120     # lifecycle extension
MIN_LIFETIME_MIN = 30  # minimum storm lifetime

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def haversine_km(p1, p2):
    return haversine(p1, p2, unit=Unit.KILOMETERS)


# ------------------------------------------------------------
# CORE TRACKING FUNCTION
# ------------------------------------------------------------
def build_storm_trajectories(df):
    """
    Build global storm trajectories from clustered events.
    """

    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    df = df.sort_values(TIME_COL)

    storm_id_counter = 0
    active_storms = {}  # storm_id -> dict
    assignments = []

    unique_times = df[TIME_COL].unique()

    for t in unique_times:
        df_t = df[df[TIME_COL] == t]

        for _, row in df_t.iterrows():
            lat, lon = row[LAT_COL], row[LON_COL]
            etype = row[TYPE_COL]

            best_match = None
            best_dist = np.inf
            best_dt = None

            for dt in range(TIME_STEP_MIN, MAX_BACKWARD_MIN + 1, TIME_STEP_MIN):
                t_prev = t - timedelta(minutes=dt)

                for sid, storm in active_storms.items():
                    if storm["last_time"] != t_prev:
                        continue

                    dist = haversine_km(
                        (lat, lon),
                        (storm["lat"], storm["lon"])
                    )

                    speed = dist / (dt / 60)

                    if dist < best_dist and dist <= MAX_DIST_KM and speed <= MAX_SPEED_KMH:
                        best_match = sid
                        best_dist = dist
                        best_dt = dt

                if best_match is not None:
                    break

            if best_match is not None:
                storm = active_storms[best_match]
                storm["lat"] = lat
                storm["lon"] = lon
                storm["last_time"] = t
                storm["times"].append(t)
                storm["lats"].append(lat)
                storm["lons"].append(lon)
                storm["types"].add(etype)

                assignments.append((best_match, t, lat, lon, etype, "observed"))

            else:
                storm_id_counter += 1
                active_storms[storm_id_counter] = {
                    "lat": lat,
                    "lon": lon,
                    "last_time": t,
                    "times": [t],
                    "lats": [lat],
                    "lons": [lon],
                    "types": {etype}
                }

                assignments.append((storm_id_counter, t, lat, lon, etype, "observed"))

    return pd.DataFrame(
        assignments,
        columns=["storm_id", "time", "lat", "lon", TYPE_COL, "source"]
    )


def assign_storm_type(traj_df, type_col=TYPE_COL):
    """
    Assign storm type based on contained event types.
    """

    storm_type = (
        traj_df
        .groupby("storm_id")[type_col]
        .apply(lambda x: "MIXED" if x.nunique() > 1 else x.iloc[0])
        .rename("storm_type")
        .reset_index()
    )

    return traj_df.merge(storm_type, on="storm_id", how="left")



# ------------------------------------------------------------
# GAP FILLING + LIFECYCLE EXTENSION
# ------------------------------------------------------------
def fill_and_extend_storms(df):
    """
    Interpolate gaps and extend storms ±2h.
    """

    out = []

    for storm_id, g in df.groupby("storm_id"):
        g = g.sort_values("time")

        duration = (g["time"].max() - g["time"].min()).total_seconds() / 60
        if duration < MIN_LIFETIME_MIN:
            continue

        # resolve event type
        types = set(g[TYPE_COL])
        if len(types) > 1:
            storm_type = "MIXED"
        else:
            storm_type = list(types)[0]

        full_time = pd.date_range(
            g["time"].min(),
            g["time"].max(),
            freq=f"{TIME_STEP_MIN}min"
        )

        g2 = (
            g.set_index("time")
             .reindex(full_time)
             .reset_index()
             .rename(columns={"index": "time"})
        )

        g2["lat"] = g2["lat"].interpolate()
        g2["lon"] = g2["lon"].interpolate()
        g2["source"] = g2["source"].fillna("interpolated")
        g2["storm_id"] = storm_id
        g2[TYPE_COL] = storm_type

        # extrapolation
        dt_hours = EXTEND_MIN / 60

        if len(g) >= 2:
            vlat = (g["lat"].iloc[-1] - g["lat"].iloc[0]) / duration * 60
            vlon = (g["lon"].iloc[-1] - g["lon"].iloc[0]) / duration * 60
        else:
            vlat = vlon = 0

        pre_times = pd.date_range(
            g2["time"].min() - timedelta(minutes=EXTEND_MIN),
            g2["time"].min() - timedelta(minutes=TIME_STEP_MIN),
            freq=f"{TIME_STEP_MIN}min"
        )

        post_times = pd.date_range(
            g2["time"].max() + timedelta(minutes=TIME_STEP_MIN),
            g2["time"].max() + timedelta(minutes=EXTEND_MIN),
            freq=f"{TIME_STEP_MIN}min"
        )

        def extrapolate(times, ref_time, ref_lat, ref_lon, sign):
            rows = []
            for t in times:
                dt = (t - ref_time).total_seconds() / 3600
                rows.append({
                    "storm_id": storm_id,
                    "time": t,
                    "lat": ref_lat + sign * vlat * dt,
                    "lon": ref_lon + sign * vlon * dt,
                    TYPE_COL: storm_type,
                    "source": "extrapolated"
                })
            return rows

        out.append(g2)
        out.append(pd.DataFrame(extrapolate(
            pre_times,
            g2["time"].iloc[0],
            g2["lat"].iloc[0],
            g2["lon"].iloc[0],
            -1
        )))
        out.append(pd.DataFrame(extrapolate(
            post_times,
            g2["time"].iloc[-1],
            g2["lat"].iloc[-1],
            g2["lon"].iloc[-1],
            +1
        )))

    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_file = os.path.join(GROUPED_DIR, "events_grouped.csv")
    print(f"🌀 Reading clustered events from {input_file}...")
    df = pd.read_csv(input_file)
   
    print("🌀 Building storm trajectories...")
    tracked = build_storm_trajectories(df)
    traj_df = assign_storm_type(tracked)

    traj_df["type_transition"] = (
    traj_df
    .groupby("storm_id")[TYPE_COL]
    .transform(lambda x: x.nunique() > 1))
    

    print("🧩 Filling gaps and extending lifecycle...")
    final = fill_and_extend_storms(traj_df)

    out_file = os.path.join(OUTPUT_DIR, "storm_trajectories.csv")
    final.to_csv(out_file, index=False)

    print(f"✅ Saved trajectories to {out_file}")
