import random
from matplotlib.patches import Rectangle
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
import sys
import boto3
import logging
import xarray as xr
import io

from s3_bucket_credentials import S3_BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
from data_utils import haversine_km

path_dir = f"/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN"
basename = "merged_MSG_CMSAF"

def read_file(s3, file_name, bucket):
    """Upload a file to an S3 bucket
    :param s3: Initialized S3 client object
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :return: object if file was uploaded, else False
    """
    try:
        #with open(file_name, "rb") as f:
        obj = s3.get_object(Bucket=bucket, Key=file_name)
        #print(obj)
        myObject = obj['Body'].read()
    except ClientError as e:
        logging.error(e)
        return None
    return myObject


def initialize_s3_client():
    """
    Initialize S3 client using credentials from s3_credentials_buckets.py
    """
    # Initialize the S3 client
    s3 = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY
    )
    return s3


def plot_clusters(df, title, output_dir):
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
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)




def plot_random_days_msg_clusters(
    df,
    n_days=1,
    max_slots=10,
    time_col="time_slot",
    domain=(5, 16, 42, 51.5),   # lon_min, lon_max, lat_min, lat_max
    res_deg=0.04,
    crop_pixels=100,
    seed=42,
    output_dir=None,
    filename="msg_physical_clusters"
):
    """
    Plot clustered events on a true MSG-like 0.04° grid
    with physically correct 100x100 pixel crops.
    """
    
    s3 = initialize_s3_client()
    
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    random.seed(seed)

    lon_min, lon_max, lat_min, lat_max = domain

    # --- crop geometry (exact) ---
    crop_size_deg = crop_pixels * res_deg   # = 4.0°
    crop_half = crop_size_deg / 2            # = 2.0°

    days = sorted(df[time_col].dt.date.unique())
    days = random.sample(days, min(n_days, len(days)))

    for day in days:
        df_day = df[df[time_col].dt.date == day]
        slots = sorted(df_day[time_col].unique())[:max_slots]

        if len(slots) == 0:
            continue

        ncols = 5
        nrows = int(np.ceil(len(slots) / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.8 * ncols, 4.8 * nrows),
            subplot_kw=dict(projection=ccrs.PlateCarree()),
        )
        axes = np.atleast_1d(axes).flatten()

        for ax, slot in zip(axes, slots):
            print(f"Plotting {day} {slot}...")
            ax.set_extent(domain)
            
            #add actual MSG field from S3
            year = slot.year
            month = f"{slot.month:02d}"
            day = slot.day
            file = f"{path_dir}/{year:04d}/{month}/{basename}_{year:04d}-{month}-{day:02d}.nc"
            print(file)
            my_obj = read_file(s3, file, S3_BUCKET_NAME)
            if my_obj is not None:
                ds_day = xr.open_dataset(io.BytesIO(my_obj), engine='h5netcdf')
                #print(ds_day)
                
                #select only variable of interest
                ds_day_var = ds_day['IR_108']

                #select time nearest to slot
                #remove utc timezone for selection
                slot_naive = slot.tz_convert(None)
                ds_slot = ds_day_var.sel(time=slot_naive, method='nearest')
                #plot
                img = ds_slot.plot.imshow(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap='gray_r',
                    vmin=240,
                    vmax=300,
                )
            else:
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=":")
                ax.add_feature(cfeature.LAND, alpha=0.25)
            # --- draw true MSG grid ---
            gl = ax.gridlines(
                draw_labels=False,
                linewidth=0.3,
                color="gray",
                alpha=0.3,
            )
            gl.xlocator = plt.MultipleLocator(res_deg)
            gl.ylocator = plt.MultipleLocator(res_deg)

            slot_df = df_day[df_day[time_col] == slot]

            # --- plot events ---
            for event_type in ['PRECIP', 'HAIL']:
                color = 'blue' if event_type == 'PRECIP' else 'green'
                ax.scatter(
                    slot_df.loc[slot_df.TYPE_EVENT == event_type, "LONGITUDE"],
                    slot_df.loc[slot_df.TYPE_EVENT == event_type, "LATITUDE"],
                    s=25,
                    c=color,
                    alpha=0.7,
                    label=event_type,
                )

            # --- cluster centers ---
            centers = (
                slot_df[slot_df.cluster_id >= 0]
                .drop_duplicates("cluster_id")
            )
            #colot clusters by event type majority
            ax.scatter(
                centers.cluster_lon,
                centers.cluster_lat,
                s=140,
                marker="X",
                color=color,
                edgecolor="white",
                linewidth=1.2,
                zorder=6,
                label="Cluster center",
            )

            # --- draw physically correct crop ---
            for _, row in centers.iterrows():
                cx, cy = row.cluster_lon, row.cluster_lat

                x0 = np.clip(cx - crop_half, lon_min, lon_max - crop_size_deg)
                y0 = np.clip(cy - crop_half, lat_min, lat_max - crop_size_deg)

                rect = Rectangle(
                    (x0, y0),
                    crop_size_deg,
                    crop_size_deg,
                    linewidth=2.2,
                    edgecolor="red",
                    facecolor="none",
                    zorder=5,
                )
                ax.add_patch(rect)

            ax.set_title(
                slot.strftime(f"%Y-%m-%d %H:%M UTC"),
                fontsize=12,
                fontweight="bold",
            )

        for ax in axes[len(slots):]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4)

        fig.suptitle(
            f"{day} - MSG-like clusters",
            fontsize=17,
            fontweight="bold",
        )

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out = f"{output_dir}/{filename}_{day}.png"
            plt.savefig(out, dpi=200)
            print(f"✅ Saved {out}")

        plt.show()
        plt.close(fig)



def trajectory_diagnostics_plot(
    df_traj,
    n_days=3,
    max_slots=10,
    domain=(5, 16, 42, 51.5),
    res_deg=0.04,
    crop_pixels=100,
    seed=42,
    output_dir=None,
):
    """
    General diagnostic and plotting function for storm trajectories.

    Parameters
    ----------
    df_traj : pd.DataFrame
        Trajectory dataframe with columns:
        ['time', 'storm_id', 'lat', 'lon', 'cluster_event_type', 'source', 'storm_type', 'type_transition']
        'source' = observed / interpolated / extrapolated
    n_days : int
        Number of random days to plot
    max_slots : int
        Maximum number of time slots per day
    domain : tuple
        (lon_min, lon_max, lat_min, lat_max)
    res_deg : float
        MSG grid resolution in degrees
    crop_pixels : int
        Crop size in pixels
    seed : int
        Random seed for day selection
    output_dir : str
        Optional directory to save plots
    """
    s3 = initialize_s3_client()
    df_traj = df_traj.copy()
    
    # Ensure proper datetime
    df_traj['time'] = pd.to_datetime(df_traj['time'], utc=True)
    df_traj['date'] = df_traj['time'].dt.date

    lon_min, lon_max, lat_min, lat_max = domain
    crop_size_deg = crop_pixels * res_deg
    crop_half = crop_size_deg / 2

    np.random.seed(seed)

    # ---- Compute trajectory statistics ----
    traj_stats = []
    for sid, group in df_traj.groupby('storm_id'):
        times = group['time'].sort_values()
        lats = group['lat']
        lons = group['lon']
        sources = group['source']

        # Distances and speeds
        dists = [0] + [haversine_km(lats.iloc[i-1], lons.iloc[i-1], lats.iloc[i], lons.iloc[i]) 
                       for i in range(1, len(group))]
        dt_hours = [0] + [(times.iloc[i] - times.iloc[i-1]).total_seconds()/3600 
                          for i in range(1, len(group))]
        speeds = [d/h if h>0 else 0 for d,h in zip(dists, dt_hours)]

        # Determine storm_type: mixed if multiple types in trajectory
        storm_types = group['storm_type'].dropna().unique()
        storm_type_final = storm_types[0] if len(storm_types)==1 else 'MIXED'

        stats = {
            'storm_id': sid,
            'n_points': len(group),
            'n_observed': (sources=='observed').sum(),
            'n_interpolated': (sources=='interpolated').sum(),
            'n_extrapolated': (sources=='extrapolated').sum(),
            'total_distance_km': np.sum(dists),
            'mean_speed_kmh': np.mean(speeds),
            'max_speed_kmh': np.max(speeds),
            'storm_type': storm_type_final,
        }
        traj_stats.append(stats)

    traj_stats_df = pd.DataFrame(traj_stats)
    print("Trajectory statistics summary:")
    print(traj_stats_df.describe(include='all'))

    # ---- Randomly select days to plot ----
    days = sorted(df_traj['date'].unique())
    plot_days = np.random.choice(days, size=min(n_days, len(days)), replace=False)

    for day in plot_days:
        df_day = df_traj[df_traj['date']==day]
        slots = sorted(df_day['time'].unique())[:max_slots]
        if len(slots) == 0:
            continue

        ncols = 5
        nrows = int(np.ceil(len(slots)/ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.8*ncols, 4.8*nrows),
            subplot_kw=dict(projection=ccrs.PlateCarree())
        )
        axes = np.atleast_1d(axes).flatten()

        for ax, slot in zip(axes, slots):
            print(f"Plotting {day} {slot}...")
            ax.set_extent(domain)
            
            #add actual MSG field from S3
            year = slot.year
            month = f"{slot.month:02d}"
            day = slot.day
            file = f"{path_dir}/{year:04d}/{month}/{basename}_{year:04d}-{month}-{day:02d}.nc"
            print(file)
            my_obj = read_file(s3, file, S3_BUCKET_NAME)
            if my_obj is not None:
                ds_day = xr.open_dataset(io.BytesIO(my_obj), engine='h5netcdf')
                #print(ds_day)
                
                #select only variable of interest
                ds_day_var = ds_day['IR_108']

                #select time nearest to slot
                #remove utc timezone for selection
                slot_naive = slot.tz_convert(None)
                ds_slot = ds_day_var.sel(time=slot_naive, method='nearest')
                #plot
                img = ds_slot.plot.imshow(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap='gray_r',
                    vmin=240,
                    vmax=300,
                )
            else:
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=":")
                ax.add_feature(cfeature.LAND, alpha=0.25)
            ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.3)

            df_slot = df_day[df_day['time']==slot]

            # --- plot events by type ---
            for event_type, color in zip(['PRECIP','HAIL','MIXED'], ['blue','green','orange']):
                ax.scatter(
                    df_slot.loc[df_slot['cluster_event_type']==event_type, 'lon'],
                    df_slot.loc[df_slot['cluster_event_type']==event_type, 'lat'],
                    s=25, c=color, alpha=0.7,
                    label=f"{event_type} events"
                )

            # --- plot cluster centers ---
            centers = df_slot.groupby('storm_id').first().reset_index()
            ax.scatter(
                centers['lon'], centers['lat'],
                s=140, marker='X', color='red', edgecolor='white', linewidth=1.2,
                zorder=6, label='Cluster center'
            )

            # --- draw crop rectangle ---
            for _, row in centers.iterrows():
                cx, cy = row['lon'], row['lat']
                x0 = np.clip(cx - crop_half, lon_min, lon_max - crop_size_deg)
                y0 = np.clip(cy - crop_half, lat_min, lat_max - crop_size_deg)
                rect = Rectangle((x0,y0), crop_size_deg, crop_size_deg,
                                 linewidth=2.2, edgecolor='red', facecolor='none', zorder=5)
                ax.add_patch(rect)

            ax.set_title(slot.strftime("%Y-%m-%d %H:%M UTC"), fontsize=12, fontweight='bold')

        for ax in axes[len(slots):]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4)
        fig.suptitle(f"{day} - Trajectory clusters", fontsize=17, fontweight='bold')
        plt.tight_layout(rect=[0,0.06,1,0.95])

        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            outpath = f"{output_dir}/trajectory_plot_{day}.png"
            plt.savefig(outpath, dpi=200)
            print(f"✅ Saved {outpath}")

        plt.show()
        plt.close(fig)

    return traj_stats_df