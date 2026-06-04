import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def filter_by_event_years(df, years):
    """
    Filter dataframe keeping only rows whose TIME_EVENT year
    is in the provided list of years.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a TIME_EVENT column
    years : list or set of int
        Years to keep (e.g. [2018, 2019, 2020])

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    df = df.copy()
    
    # Convert TIME_EVENT to datetime (ISO format with Z is supported)
    df["TIME_EVENT"] = pd.to_datetime(df["TIME_EVENT"], errors="coerce")
    
    # Filter by year
    df_filtered = df[df["TIME_EVENT"].dt.year.isin(years)]
    
    return df_filtered


def add_msg_time_slot(df, time_col="TIME_EVENT", freq="15min"):
    """
    Assign each event to a MSG-aligned time slot.

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str
        Column containing event timestamps
    freq : str
        Pandas offset alias (e.g. "15min", "30min", "1H")

    Examples
    --------
    freq="15min" → 10:07 → 10:00
    freq="30min" → 10:22 → 10:00
    freq="1H"    → 10:45 → 10:00
    """
    df = df.copy()

    # Ensure datetime
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    # Align to MSG slots
    df["time_slot"] = df[time_col].dt.floor(freq)

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


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))



def track_clusters_over_time(
    df,
    time_col="time_slot",
    cluster_id_col="cluster_id",
    lat_col="cluster_lat",
    lon_col="cluster_lon",
    max_dist_km=150,
):
    """
    Assign persistent storm_id to clusters evolving in time.
    """

    df = df.sort_values(time_col).copy()
    df["storm_id"] = -1

    next_storm_id = 0
    prev_clusters = {}

    for t in sorted(df[time_col].unique()):
        current = df[df[time_col] == t]

        for idx, row in current.iterrows():
            best_id = None
            best_dist = np.inf

            for sid, prev in prev_clusters.items():
                d = haversine_km(
                    row[lat_col], row[lon_col],
                    prev[lat_col], prev[lon_col]
                )
                if d < best_dist and d <= max_dist_km:
                    best_dist = d
                    best_id = sid

            if best_id is None:
                df.loc[idx, "storm_id"] = next_storm_id
                next_storm_id += 1
            else:
                df.loc[idx, "storm_id"] = best_id

        # update memory
        prev_clusters = (
            df[df[time_col] == t]
            .set_index("storm_id")
            [[lat_col, lon_col]]
            .to_dict("index")
        )

    return df
