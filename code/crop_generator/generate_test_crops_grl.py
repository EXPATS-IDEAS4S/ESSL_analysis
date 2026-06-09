"""
Generate GRL 2026 test crops from MSG satellite data for the ESSL case list.

For each case in the input case CSV, the script reads one daily MSG NetCDF file,
selects the case time window, and creates 100 x 100 pixel crop sequences in two
modes:

1. Lagrangian mode
   - One moving crop sequence per case.
   - The crop follows the case from start_lat/start_lon to end_lat/end_lon.
   - If the start and end report locations fit inside the first crop, the crop
     center remains fixed at the start location.

2. Eulerian mode
   - A fixed grid of numbered crop views covers as much of the EXPATS domain as
     possible: lon 5-16, lat 42-51.5.
   - Each numbered view is cropped at every timestamp.
   - The view number is plotted on the eulerian map so each video can be traced
     back to its domain location.

Processed variables
-------------------
- IR_108: raw 10.8 um brightness temperature.
- cma: cloud mask.
- IR_108_masked: cloud-mask-applied copy of IR_108. Non-cloud pixels are set to
  the IR_108 maximum value and appear black in the MP4 videos.

Input files
-----------
- Case list:
  /sat_data/output/grl_2026/csv/essl_cases_2025_grl.csv

- ESWD reports, used only for map overlays:
  /sat_data/output/grl_2026/csv/eswd-v2-2012-2025_expats.csv

- DEM/orography for map contours:
  /data1/DEM_EXPATS_0.01x0.01.nc

- Daily MSG satellite files, read from bucket expats-msg-training:
  /data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN/YYYY/MM/
  merged_MSG_CMSAF_YYYY-MM-DD.nc

Output files
------------
Base output directory:
  /sat_data/crops/GRL_testing_crops

NetCDF chunks are saved with data variables on fixed dimensions
(time, y, x), where y = 100 and x = 100. Geographic coordinates are stored as
lat(time, y), lon(time, x), plus crop_center_lat(time) and crop_center_lon(time).

Output filename pattern:
  YYYYMMDD_HHMM-HHMM_<mode>[_viewNNN]_chunkNNN_nNN_IR_108

Lagrangian NetCDF:
  /sat_data/crops/GRL_testing_crops/YYYY-MM-DD/nc/lagrangian/
  YYYYMMDD_HHMM-HHMM_lagrangian_chunkNNN_nNN_IR_108.nc

Eulerian NetCDF:
  /sat_data/crops/GRL_testing_crops/YYYY-MM-DD/nc/eulerian/view_NNN/
  YYYYMMDD_HHMM-HHMM_eulerian_viewNNN_chunkNNN_nNN_IR_108.nc

MP4 videos:
  Only IR_108_masked videos are saved. Display range is 240-320 K; masked
  non-cloud pixels are black.

Lagrangian videos and maps:
  /sat_data/crops/GRL_testing_crops/YYYY-MM-DD/videos/lagrangian/IR_108_masked/
  sequence/mp4_vmin-vmax_greyscale_CMA/

Eulerian videos:
  /sat_data/crops/GRL_testing_crops/YYYY-MM-DD/videos/eulerian/IR_108_masked/
  view_NNN/mp4_vmin-vmax_greyscale_CMA/

Eulerian maps:
  /sat_data/crops/GRL_testing_crops/YYYY-MM-DD/videos/eulerian/maps/view_NNN/
  mp4_vmin-vmax_greyscale_CMA/

Maps include the raw IR_108 first frame for the sequence/view, BT108 colorbar,
DEM/orography contours up to 4000 m, ESWD reports, and crop footprints.

author: Claudia Acquistapace
date: 2026-06-09
"""

import os
import io
import boto3
import xarray as xr
import sys
import pandas as pd 
import yaml
import numpy as np
import subprocess
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.append('/home/claudia/codes/ESSL_analysis/code/')

from cropping_functions import crops_by_center, apply_cma_mask
from credentials_buckets import S3_BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
import aux_func

MSG_BUCKET_NAME = "expats-msg-training"
DEM_PATH = "/data1/DEM_EXPATS_0.01x0.01.nc"
BT108_DISPLAY_MIN = 240.0
BT108_DISPLAY_MAX = 320.0
OROGRAPHY_VARIABLE_CANDIDATES = (
    'orography', 'orog', 'orography_m', 'elevation', 'altitude',
    'surface_altitude', 'HSURF', 'height', 'dem', 'DEM'
)


def parse_utc_time_as_naive(timestamp):
    """Return a timezone-naive UTC timestamp for xarray datetime64 slicing."""
    return pd.to_datetime(timestamp, utc=True).tz_convert(None)



def define_lagrangian_crop_centers(
    ds_day_var, x_pixel, y_pixel,
    lat_start, lon_start, lat_end, lon_end,
    num_timestamps
):
    """"
    Define crop centers for the lagrangian crops, based on the start and end location of the case
        and the number of timestamps in the data.
        -  If start and end centers are both in the frame centered on lat start lon start, 
        then we can generate crops centered on the start location for all timestamps.
        -  If start and end centers are not in the frame centered on lat start lon start, 
        then we need to generate crops centered as follows: 
            - the first and last timestamp will be used to generate crops centered on the start and end location of the case, respectively. 
            - the other timestamps will be used to generate crops centered on the intermediate locations of the case, by interpolating linearly between the start and end location of the case.
    Input:
    - ds_day_var: xarray dataset containing the satellite data for the day of the case
    - x_pixel, y_pixel: size of the crops in pixels
    - lat_start, lon_start: latitude and longitude of the start location of the case
    - lat_end, lon_end: latitude and longitude of the end location of the case
    - num_timestamps: number of timestamps in the data

    Output:
    - crop_centers: list of tuples (lat, lon) of the crop centers for each timestamp in the data

    """

    # define crop centered in the start location
    start_crop = crops_by_center(ds_day_var.isel(time=0), x_pixel, y_pixel, (lat_start, lon_start))

    # check if the end location is in the frame centered on the start location
    if (lat_end >= start_crop.lat.min() and lat_end <= start_crop.lat.max() and
        lon_end >= start_crop.lon.min() and lon_end <= start_crop.lon.max()):   
        # if the end location is in the frame centered on the start location, we can generate crops centered on the start location for all timestamps
        crop_centers = [(lat_start, lon_start)] * num_timestamps
    else:
        # if the end location is not in the frame centered on the start location, we need to generate crops centered as follows: 
        # - the first and last timestamp will be used to generate crops centered on the start and end location of the case, respectively. 
        # - the other timestamps will be used to generate crops centered on the intermediate locations of the case, by interpolating
        # linearly between the start and end location of the case.

        crop_centers = []
        for i in range(num_timestamps):
            if i == 0:
                crop_centers.append((lat_start, lon_start))
            elif i == num_timestamps - 1:
                crop_centers.append((lat_end, lon_end))
            else:
                lat_i = lat_start + (lat_end - lat_start) * i / (num_timestamps - 1)
                lon_i = lon_start + (lon_end - lon_start) * i / (num_timestamps - 1)
                crop_centers.append((lat_i, lon_i))

    return crop_centers


def tile_start_indices(idx_min, idx_max, crop_size, n_points):
    """
    Return crop start indices that cover a coordinate-index interval as much as possible.
    """
    if crop_size > n_points:
        raise ValueError(f"Crop size {crop_size} is larger than grid size {n_points}")

    start_min = max(0, idx_min)
    start_max = min(n_points - crop_size, idx_max - crop_size + 1)

    if start_max < start_min:
        center_idx = (idx_min + idx_max) // 2
        start = int(np.clip(center_idx - crop_size // 2, 0, n_points - crop_size))
        return [start]

    starts = list(range(start_min, start_max + 1, crop_size))
    if starts[-1] != start_max:
        starts.append(start_max)

    return starts


def define_eulerian_crop_centers(ds_image, x_pixel, y_pixel, domain):
    """
    Define numbered eulerian crop centers that tile the EXPATS domain.

    Parameters
    ----------
    ds_image : xarray.Dataset or xarray.DataArray
        Dataset containing `lat` and `lon` coordinates.
    x_pixel, y_pixel : int
        Crop size in pixels.
    domain : tuple
        Domain as (lon_min, lon_max, lat_min, lat_max).

    Returns
    -------
    crop_specs : list of dict
        One dictionary per crop, with view_id, grid row/column, center lat/lon,
        and crop bounds. Use (center_lat, center_lon) with crops_by_center().
    """
    lon_min, lon_max, lat_min, lat_max = domain

    lon_values = ds_image.lon.values
    lat_values = ds_image.lat.values

    lon_indices = np.where((lon_values >= lon_min) & (lon_values <= lon_max))[0]
    lat_indices = np.where((lat_values >= lat_min) & (lat_values <= lat_max))[0]

    if len(lon_indices) == 0 or len(lat_indices) == 0:
        raise ValueError("Requested eulerian domain does not overlap dataset lat/lon coordinates")

    lon_starts = tile_start_indices(
        int(lon_indices.min()), int(lon_indices.max()), x_pixel, len(lon_values)
    )
    lat_starts = tile_start_indices(
        int(lat_indices.min()), int(lat_indices.max()), y_pixel, len(lat_values)
    )

    crop_specs = []
    view_id = 1

    for row, y_start in enumerate(lat_starts):
        for col, x_start in enumerate(lon_starts):
            y_end = y_start + y_pixel
            x_end = x_start + x_pixel
            y_center = y_start + y_pixel // 2
            x_center = x_start + x_pixel // 2

            crop_lats = lat_values[y_start:y_end]
            crop_lons = lon_values[x_start:x_end]

            crop_specs.append({
                'view_id': view_id,
                'row': row,
                'col': col,
                'center_lat': float(lat_values[y_center]),
                'center_lon': float(lon_values[x_center]),
                'lat_min': float(np.min(crop_lats)),
                'lat_max': float(np.max(crop_lats)),
                'lon_min': float(np.min(crop_lons)),
                'lon_max': float(np.max(crop_lons)),
                'y_start': int(y_start),
                'y_end': int(y_end),
                'x_start': int(x_start),
                'x_end': int(x_end),
            })
            view_id += 1

    return crop_specs


def add_cloud_mask_and_masked_ir108(ds_t, ds_var_t, values_max, var_names):
    """
    Add a CMA-masked copy of IR_108 while preserving the raw IR_108 and cma fields.
    """
    if 'cma' not in ds_t or 'IR_108' not in ds_var_t:
        return ds_var_t

    ds_var_t = ds_var_t.copy()
    ir_108_max = values_max[var_names.index('IR_108')]
    ds_ir_masked = apply_cma_mask(ds_t, ds_var_t[['IR_108']].copy(), [ir_108_max])
    ds_var_t['IR_108_masked'] = ds_ir_masked['IR_108']

    return ds_var_t


def data_array_to_video_frames(ds_image, vmin, vmax, reverse=True):
    """Convert a time-indexed crop into grayscale uint8 frames for a video."""
    frames = []

    if 'time' in ds_image.dims:
        image_slices = [ds_image.sel(time=t) for t in ds_image['time'].values]
    else:
        image_slices = [ds_image]

    for ds_time in image_slices:
        data = ds_time.values.squeeze()
        data = np.flipud(data)
        data = np.clip(data, vmin, vmax)
        data = np.nan_to_num(data, nan=vmax, posinf=vmax, neginf=vmin)

        if vmax > vmin:
            data = (data - vmin) / (vmax - vmin) * 255.0
        else:
            data = np.zeros_like(data)

        if reverse:
            data = 255.0 - data

        frames.append(data.astype(np.uint8))

    return frames


def crop_chunk_to_video_frames(crop_chunk, var_name, vmin, vmax, reverse=True):
    """Convert per-timestamp crops into same-sized grayscale video frames."""
    frames = []

    for ds_crop in crop_chunk:
        if var_name not in ds_crop:
            continue

        data = ds_crop[var_name].values.squeeze()
        data = np.flipud(data)
        data = np.clip(data, vmin, vmax)
        data = np.nan_to_num(data, nan=vmax, posinf=vmax, neginf=vmin)

        if vmax > vmin:
            data = (data - vmin) / (vmax - vmin) * 255.0
        else:
            data = np.zeros_like(data)

        if reverse:
            data = 255.0 - data

        frames.append(data.astype(np.uint8))

    return frames


def save_crop_chunk_video(crop_chunk, var_name, filename, out_path, vmin, vmax, fps=5, reverse=True):
    """Save a short MP4 video from unaligned per-timestamp crop frames."""
    frames = crop_chunk_to_video_frames(crop_chunk, var_name, vmin, vmax, reverse=reverse)
    if not frames:
        return

    os.makedirs(out_path, exist_ok=True)
    video_path = os.path.join(out_path, f"{filename}.mp4")
    height, width = frames[0].shape
    frame_bytes = b''.join(frame.tobytes() for frame in frames)

    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'gray',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-c:v', 'libx264',
        '-crf', '28',
        '-preset', 'veryfast',
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        video_path,
    ]
    subprocess.run(cmd, input=frame_bytes, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Saved video to {video_path}")


def save_crop_video(ds_image, filename, out_path, vmin, vmax, fps=5, reverse=True):
    """Save a short MP4 video for a crop chunk."""
    frames = data_array_to_video_frames(ds_image, vmin, vmax, reverse=reverse)
    if not frames:
        return

    os.makedirs(out_path, exist_ok=True)
    video_path = os.path.join(out_path, f"{filename}.mp4")
    height, width = frames[0].shape
    frame_bytes = b''.join(frame.tobytes() for frame in frames)

    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'gray',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-c:v', 'libx264',
        '-crf', '28',
        '-preset', 'veryfast',
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        video_path,
    ]
    subprocess.run(cmd, input=frame_bytes, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Saved video to {video_path}")


def interpolate_rgb(start_color, end_color, fraction):
    """Interpolate between two RGB colors."""
    return tuple(
        int(start + (end - start) * fraction)
        for start, end in zip(start_color, end_color)
    )


def lonlat_to_pixel(lon, lat, domain, map_width, map_height, margin):
    """Project lon/lat to image pixels using a simple rectangular lon/lat map."""
    lon_min, lon_max, lat_min, lat_max = domain
    x = margin + (lon - lon_min) / (lon_max - lon_min) * (map_width - 2 * margin)
    y = map_height - margin - (lat - lat_min) / (lat_max - lat_min) * (map_height - 2 * margin)
    return int(round(x)), int(round(y))


def find_orography_field(ds):
    """Return the first orography-like field in a dataset, if one is available."""
    lower_name_map = {name.lower(): name for name in ds.data_vars}

    for candidate in OROGRAPHY_VARIABLE_CANDIDATES:
        var_name = lower_name_map.get(candidate.lower())
        if var_name and {'lat', 'lon'} <= set(ds[var_name].dims):
            da = ds[var_name]
            if 'time' in da.dims:
                da = da.isel(time=0)
            return da

    for var_name, da in ds.data_vars.items():
        standard_name = str(da.attrs.get('standard_name', '')).lower()
        long_name = str(da.attrs.get('long_name', '')).lower()
        if {'lat', 'lon'} <= set(da.dims) and ('altitude' in standard_name or 'orograph' in long_name):
            if 'time' in da.dims:
                da = da.isel(time=0)
            return da

    return None


def load_orography_from_dem(path):
    """Load orography from a DEM NetCDF file."""
    if not os.path.exists(path):
        print(f"DEM file not found at {path}; orography overlay will be skipped.")
        return None

    try:
        ds_dem = xr.open_dataset(path)
    except Exception as exc:
        print(f"Could not open DEM file {path}: {exc}; orography overlay will be skipped.")
        return None

    orography = find_orography_field(ds_dem)
    if orography is not None:
        return orography

    for var_name, da in ds_dem.data_vars.items():
        if {'lat', 'lon'} <= set(da.dims):
            print(f"Using {var_name} from {path} as orography field.")
            if 'time' in da.dims:
                da = da.isel(time=0)
            return da

    print(f"No lat/lon DEM field found in {path}; orography overlay will be skipped.")
    return None


def draw_orography(image, domain, map_width, map_height, margin, orography):
    """Draw an orography field as a light grayscale background."""
    if orography is None or not {'lat', 'lon'} <= set(orography.dims):
        return

    lon_min, lon_max, lat_min, lat_max = domain
    oro = orography.where(
        (orography.lon >= lon_min) & (orography.lon <= lon_max) &
        (orography.lat >= lat_min) & (orography.lat <= lat_max),
        drop=True
    )
    if oro.size == 0:
        return

    data = oro.values.squeeze()
    if data.ndim != 2 or np.all(np.isnan(data)):
        return

    data = np.nan_to_num(data, nan=np.nanmin(data))
    data_min = np.nanpercentile(data, 2)
    data_max = np.nanpercentile(data, 98)
    data = np.clip(data, data_min, data_max)
    if data_max > data_min:
        data = (data - data_min) / (data_max - data_min)
    else:
        data = np.zeros_like(data)

    data = 245 - (data * 95)
    if float(oro.lat.values[0]) < float(oro.lat.values[-1]):
        data = np.flipud(data)

    domain_width = map_width - 2 * margin
    domain_height = map_height - 2 * margin
    oro_image = Image.fromarray(data.astype(np.uint8), mode='L').resize((domain_width, domain_height))
    oro_rgb = Image.merge('RGB', (oro_image, oro_image, oro_image))
    image.paste(oro_rgb, (margin, margin))


def draw_reports(draw, reports_df, domain, map_width, map_height, margin):
    """Draw ESWD reports on the map, using marker shape/color by event type."""
    if reports_df is None or reports_df.empty:
        return

    lon_min, lon_max, lat_min, lat_max = domain
    reports = reports_df[
        (reports_df['LONGITUDE'] >= lon_min) & (reports_df['LONGITUDE'] <= lon_max) &
        (reports_df['LATITUDE'] >= lat_min) & (reports_df['LATITUDE'] <= lat_max)
    ]
    if reports.empty:
        return

    for _, report in reports.iterrows():
        x, y = lonlat_to_pixel(report['LONGITUDE'], report['LATITUDE'], domain, map_width, map_height, margin)
        event_type = str(report.get('TYPE_EVENT', '')).upper()

        if event_type == 'HAIL':
            color = (225, 118, 28)
            points = [(x, y - 6), (x - 6, y + 5), (x + 6, y + 5)]
            draw.polygon(points, fill=color, outline=(40, 40, 40))
        elif event_type == 'PRECIP':
            color = (43, 124, 210)
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color, outline=(40, 40, 40))
        else:
            color = (90, 90, 90)
            draw.rectangle([x - 4, y - 4, x + 4, y + 4], fill=color, outline=(40, 40, 40))


def load_reports_for_case(reports_df, case_date, start_time, end_time):
    """Filter the full ESWD report table to the current case date and time window."""
    if reports_df is None or reports_df.empty:
        return None

    reports = reports_df[reports_df['date'] == case_date.normalize()].copy()
    if reports.empty:
        return reports

    reports['time_event_naive'] = pd.to_datetime(reports['TIME_EVENT'], utc=True).dt.tz_convert(None)
    return reports[
        (reports['time_event_naive'] >= start_time) &
        (reports['time_event_naive'] <= end_time)
    ]


def plot_orography_contours(ax, orography, domain):
    """Overlay DEM orography as contour lines."""
    if orography is None or not {'lat', 'lon'} <= set(orography.dims):
        return

    lon_min, lon_max, lat_min, lat_max = domain
    oro = orography.where(
        (orography.lon >= lon_min) & (orography.lon <= lon_max) &
        (orography.lat >= lat_min) & (orography.lat <= lat_max),
        drop=True
    )
    if oro.size == 0:
        return

    data = oro.values.squeeze()
    if data.ndim != 2 or np.all(np.isnan(data)):
        return

    levels = np.arange(0, 4001, 500)
    levels = levels[(levels >= np.nanmin(data)) & (levels <= max(4000, np.nanmax(data)))]
    if len(levels) < 2:
        return

    contours = ax.contour(
        oro.lon.values, oro.lat.values, data,
        levels=levels, colors='0.35', linewidths=0.7, alpha=0.65
    )
    ax.clabel(contours, inline=True, fontsize=7, fmt='%d m')


def plot_reports_on_axis(ax, reports_df, domain):
    """Overlay ESWD reports on a matplotlib lon/lat axis."""
    if reports_df is None or reports_df.empty:
        return

    lon_min, lon_max, lat_min, lat_max = domain
    reports = reports_df[
        (reports_df['LONGITUDE'] >= lon_min) & (reports_df['LONGITUDE'] <= lon_max) &
        (reports_df['LATITUDE'] >= lat_min) & (reports_df['LATITUDE'] <= lat_max)
    ]
    if reports.empty:
        return

    precip = reports[reports['TYPE_EVENT'].astype(str).str.upper() == 'PRECIP']
    hail = reports[reports['TYPE_EVENT'].astype(str).str.upper() == 'HAIL']
    other = reports[~reports.index.isin(precip.index.union(hail.index))]

    if not precip.empty:
        ax.scatter(precip['LONGITUDE'], precip['LATITUDE'], s=18, c='#2b7cd2',
                   marker='o', edgecolors='black', linewidths=0.3, label='PRECIP')
    if not hail.empty:
        ax.scatter(hail['LONGITUDE'], hail['LATITUDE'], s=28, c='#e1761c',
                   marker='^', edgecolors='black', linewidths=0.3, label='HAIL')
    if not other.empty:
        ax.scatter(other['LONGITUDE'], other['LATITUDE'], s=14, c='0.45',
                   marker='s', edgecolors='black', linewidths=0.3, label='Other report')


def save_crop_sequence_map(
    ds_chunk, crop_chunk, filename, out_path, domain, orography=None, reports_df=None
):
    """Save one map for a chunk with 10.8 um, DEM contours, reports, and crop footprints."""
    os.makedirs(out_path, exist_ok=True)
    map_path = os.path.join(out_path, f"{filename}_map.png")
    lon_min, lon_max, lat_min, lat_max = domain

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    #ax.set_title("EXPATS domain: IR 10.8, DEM contours, reports, and lagrangian crop footprints")
    ax.grid(True, color='0.88', linewidth=0.8)

    ir_mesh = None
    if 'IR_108' in ds_chunk:
        ir_first = ds_chunk['IR_108'].isel(time=0)
        ir_mesh = ax.pcolormesh(
            ir_first.lon.values,
            ir_first.lat.values,
            ir_first.values.squeeze(),
            cmap='gray_r',
            vmin=BT108_DISPLAY_MIN,
            vmax=BT108_DISPLAY_MAX,
            shading='auto',
            alpha=0.85,
            zorder=1,
        )
        cbar = fig.colorbar(ir_mesh, ax=ax, pad=0.02, shrink=0.82)
        cbar.set_label("BT108 (K)")

    plot_orography_contours(ax, orography, domain)

    start_color = (96, 38, 158)     # purple
    end_color = (118, 215, 234)     # light blue
    n_crops = len(crop_chunk)

    for index, ds_crop in enumerate(crop_chunk):
        fraction = index / (n_crops - 1) if n_crops > 1 else 0
        color = interpolate_rgb(start_color, end_color, fraction)

        crop_lon_min = float(ds_crop.lon.min())
        crop_lon_max = float(ds_crop.lon.max())
        crop_lat_min = float(ds_crop.lat.min())
        crop_lat_max = float(ds_crop.lat.max())

        color_hex = '#%02x%02x%02x' % color
        rect = Rectangle(
            (crop_lon_min, crop_lat_min),
            crop_lon_max - crop_lon_min,
            crop_lat_max - crop_lat_min,
            edgecolor=color_hex,
            facecolor='none',
            linewidth=2.0,
            zorder=4,
        )
        ax.add_patch(rect)
        ax.plot(
            (crop_lon_min + crop_lon_max) / 2,
            (crop_lat_min + crop_lat_max) / 2,
            marker='o',
            color=color_hex,
            markersize=4,
            zorder=5,
        )

    plot_reports_on_axis(ax, reports_df, domain)

    ax.plot([], [], color='#60269e', linewidth=2, label='First crop')
    ax.plot([], [], color='#76d7ea', linewidth=2, label='Last crop')
    ax.plot([], [], color='0.35', linewidth=1, label='orography')
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(map_path, dpi=150)
    plt.close(fig)
    print(f"Saved crop sequence map to {map_path}")


def save_eulerian_views_map(
    ds_chunk, crop_specs, current_spec, filename, out_path, domain,
    orography=None, reports_df=None
):
    """Save a domain map with the current IR_108 view and all eulerian crop views numbered."""
    os.makedirs(out_path, exist_ok=True)
    map_path = os.path.join(out_path, f"{filename}_map.png")
    lon_min, lon_max, lat_min, lat_max = domain

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color='0.88', linewidth=0.8)

    if 'IR_108' in ds_chunk:
        ir_first = ds_chunk['IR_108'].isel(time=0)
        lat_first = ds_chunk['lat'].isel(time=0)
        lon_first = ds_chunk['lon'].isel(time=0)
        ir_mesh = ax.pcolormesh(
            lon_first.values,
            lat_first.values,
            ir_first.values.squeeze(),
            cmap='gray_r',
            vmin=BT108_DISPLAY_MIN,
            vmax=BT108_DISPLAY_MAX,
            shading='auto',
            alpha=0.85,
            zorder=1,
        )
        cbar = fig.colorbar(ir_mesh, ax=ax, pad=0.02, shrink=0.82)
        cbar.set_label("BT108 (K)")

    plot_orography_contours(ax, orography, domain)
    plot_reports_on_axis(ax, reports_df, domain)

    for spec in crop_specs:
        is_current = spec['view_id'] == current_spec['view_id']
        edge_color = '#d62728' if is_current else '0.35'
        line_width = 2.4 if is_current else 1.0
        zorder = 5 if is_current else 3

        rect = Rectangle(
            (spec['lon_min'], spec['lat_min']),
            spec['lon_max'] - spec['lon_min'],
            spec['lat_max'] - spec['lat_min'],
            edgecolor=edge_color,
            facecolor='none',
            linewidth=line_width,
            zorder=zorder,
        )
        ax.add_patch(rect)
        ax.text(
            spec['center_lon'], spec['center_lat'], str(spec['view_id']),
            color=edge_color, ha='center', va='center',
            fontsize=8 if not is_current else 10,
            fontweight='bold' if is_current else 'normal',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.65, 'pad': 1.2},
            zorder=zorder + 1,
        )

    ax.plot([], [], color='0.35', linewidth=1, label='Eulerian views')
    ax.plot([], [], color='#d62728', linewidth=2.4, label=f"Current view {current_spec['view_id']}")
    ax.plot([], [], color='0.35', linewidth=1, label='orography')
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(map_path, dpi=150)
    plt.close(fig)
    print(f"Saved eulerian views map to {map_path}")


def crop_chunk_to_pixel_dataset(crop_chunk):
    """
    Convert moving lagrangian crops to a fixed pixel grid.

    Each input crop keeps its own lat/lon coordinates. The data variables are
    saved on fixed (time, y, x) dimensions so the NetCDF stays 100 x 100 instead
    of expanding to the union of moving geographic coordinates.
    """
    if not crop_chunk:
        return xr.Dataset()

    first_crop = crop_chunk[0]
    n_y = len(first_crop.lat)
    n_x = len(first_crop.lon)
    times = []
    lat_values = []
    lon_values = []
    data_vars = {}

    for ds_crop in crop_chunk:
        if len(ds_crop.lat) != n_y or len(ds_crop.lon) != n_x:
            raise ValueError(
                "All crops in a chunk must have the same pixel dimensions; "
                f"expected {(n_y, n_x)}, got {(len(ds_crop.lat), len(ds_crop.lon))}."
            )

        if 'time' in ds_crop.coords:
            times.append(ds_crop.time.values[0])
        else:
            times.append(np.datetime64('NaT'))

        lat_values.append(ds_crop.lat.values)
        lon_values.append(ds_crop.lon.values)

    for var_name in first_crop.data_vars:
        data_vars[var_name] = (
            ('time', 'y', 'x'),
            np.stack([
                ds_crop[var_name].values.squeeze()
                for ds_crop in crop_chunk
            ], axis=0),
            first_crop[var_name].attrs,
        )

    ds_chunk = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': np.asarray(times),
            'y': np.arange(n_y),
            'x': np.arange(n_x),
            'lat': (('time', 'y'), np.stack(lat_values, axis=0)),
            'lon': (('time', 'x'), np.stack(lon_values, axis=0)),
        },
    )
    ds_chunk['crop_center_lat'] = ('time', np.mean(np.stack(lat_values, axis=0), axis=1))
    ds_chunk['crop_center_lon'] = ('time', np.mean(np.stack(lon_values, axis=0), axis=1))
    ds_chunk.attrs['grid'] = 'lagrangian pixel grid; lat/lon vary by time'

    return ds_chunk


def save_crop_chunk(
    crop_chunk, chunk_start_index, outpath_crops, outpath_img, date, file_extension,
    var_names, var_props, values_min, values_max, save_video, apply_cma, expats_domain,
    orography=None, reports_df=None, mode='lagrangian', view_spec=None,
    eulerian_crop_specs=None, video_vars=None
):
    """
    Save a list of per-timestamp crops as one NetCDF file.
    """
    if not crop_chunk:
        return

    ds_chunk = crop_chunk_to_pixel_dataset(crop_chunk)
    first_time = pd.to_datetime(ds_chunk.time.values[0])
    last_time = pd.to_datetime(ds_chunk.time.values[-1])
    date_str = first_time.strftime("%Y%m%d")
    start_time_str = first_time.strftime("%H%M")
    end_time_str = last_time.strftime("%H%M")
    view_part = f"_view{view_spec['view_id']:03d}" if view_spec is not None else ""
    filename_to_save = f"{date_str}_{start_time_str}-{end_time_str}_{mode}{view_part}_chunk{chunk_start_index:03d}_n{len(crop_chunk):02d}_IR_108"

    if view_spec is not None:
        save_dir = os.path.join(outpath_crops, date, file_extension, mode, f"view_{view_spec['view_id']:03d}")
    else:
        save_dir = os.path.join(outpath_crops, date, file_extension, mode)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename_to_save}.{file_extension}")
    ds_chunk.to_netcdf(save_path)
    print(f"Saved cropped dataset chunk to {save_path}")

    if save_video:
        if video_vars is None:
            video_vars = ['IR_108_masked']

        for video_var in video_vars:
            if video_var not in ds_chunk:
                continue

            video_save_path = os.path.join(
                outpath_img, date, 'videos', mode, video_var,
                f"view_{view_spec['view_id']:03d}" if view_spec is not None else 'sequence',
                'mp4_vmin-vmax_greyscale_CMA' if apply_cma else 'mp4_vmin-vmax_greyscale'
            )
            if video_var in ['IR_108', 'IR_108_masked']:
                vmin = BT108_DISPLAY_MIN
                vmax = BT108_DISPLAY_MAX
                reverse = True
            else:
                vmin = values_min[var_names.index(video_var)]
                vmax = values_max[var_names.index(video_var)]
                reverse = False

            save_crop_chunk_video(
                crop_chunk, video_var, filename_to_save, video_save_path,
                vmin, vmax, reverse=reverse
            )

        map_base_path = os.path.join(
            outpath_img, date, 'videos', mode, 'maps',
            f"view_{view_spec['view_id']:03d}" if view_spec is not None else 'sequence'
        )
        if view_spec is None:
            save_crop_sequence_map(
                ds_chunk, crop_chunk, filename_to_save, map_base_path,
                expats_domain, orography=orography, reports_df=reports_df
            )
        elif eulerian_crop_specs is not None:
            save_eulerian_views_map(
                ds_chunk, eulerian_crop_specs, view_spec, filename_to_save,
                map_base_path, expats_domain, orography=orography,
                reports_df=reports_df
            )



def main():

    # Initialize the S3 client
    s3 = aux_func.init_s3()

    # Define variables properties
    path_to_config = "/home/claudia/codes/ESSL_analysis/code/crop_generator"
    with open(path_to_config + "/variables_config.yaml") as f:
        cfg = yaml.safe_load(f)

    #Implemented vars are IR_108, WV_062, OT
    var_names = ['IR_108', 'cma'] 

    #get variable properties from the config file
    var_props = {var: cfg['variables'][var] for var in var_names}
    #print(var_props)

    values_max = []
    values_min = []
    for var in var_names:
        value_min = var_props[var]['valid_range']['min']
        value_max = var_props[var]['valid_range']['max']
        values_min.append(value_min)
        values_max.append(value_max)
    #print(values_max, values_min)

    x_pixel = 100 
    y_pixel = 100 

    expats_domain = (5, 16, 42, 51.5) # lonmin, lonmax, latmin, latmax
    orography = load_orography_from_dem(DEM_PATH)

    apply_cma = True #if True, the cma variable will be included in the crops
    file_extension = 'nc'  # File extension for the dataset files
    save_video = True
    chunk_size = 8

    # define output path
    base_output_path = "/sat_data/crops/GRL_testing_crops"
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)
    outpath_crops = f'{base_output_path}'
    outpath_img = f'{base_output_path}'
    os.makedirs(outpath_crops, exist_ok=True)
    os.makedirs(outpath_img, exist_ok=True)


    # read list of cases from csv file
    cases_csv_path = "/sat_data/output/grl_2026/csv/essl_cases_2025_grl.csv"
    cases_df = pd.read_csv(cases_csv_path)  

    reports_csv_path = "/sat_data/output/grl_2026/csv/eswd-v2-2012-2025_expats.csv"
    if os.path.exists(reports_csv_path):
        reports_df = pd.read_csv(reports_csv_path)
        reports_df["TIME_EVENT"] = reports_df["TIME_EVENT"].astype(str)
        reports_df["date"] = pd.to_datetime(reports_df["TIME_EVENT"].str.slice(0, 10))
        print("ESWD reports CSV loaded for map overlays.")
    else:
        reports_df = None
        print(f"ESWD reports CSV not found at {reports_csv_path}; report overlays will be skipped.")


    # loop on cases and generate crops
    for index, row in cases_df.iterrows():

        case_date = pd.to_datetime(row["date"])
        date = case_date.strftime("%Y-%m-%d")
        start_time = parse_utc_time_as_naive(row["start_time"])
        end_time = parse_utc_time_as_naive(row["end_time"])
        case_reports_df = load_reports_for_case(reports_df, case_date, start_time, end_time)
        lat_start = row["start_lat"]
        lon_start = row["start_lon"]
        lat_end = row["end_lat"]
        lon_end = row["end_lon"]

        num_reports = row["num_reports"]
        num_rain_reports = row["num_precip"]
        num_hail_reports = row["num_hail"]
        duration_hours = row["duration_hours"]
        case_type = row["case_type"]

        print(f"Processing case {index+1}/{len(cases_df)}: {date} {start_time}-{end_time} {case_type} with {num_reports} reports")

        # get month, day, year from date
        year = case_date.year
        month = case_date.month
        day = case_date.day

        # define filename for the dataset files
        path_file_bucket = f"/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN/{year:04d}/{month:02d}/merged_MSG_CMSAF_{year:04d}-{month:02d}-{day:02d}.nc"
        print(f"Reading dataset file {path_file_bucket}")

        #Read file from the bucket
        my_obj = aux_func.read_file(s3, path_file_bucket, MSG_BUCKET_NAME)

        if my_obj is not None:

            # read dataset file as xarray dataset
            ds_day = xr.open_dataset(io.BytesIO(my_obj))
            #print(ds_day)
            
            #select only variable of interest
            ds_day_var = ds_day[var_names]
            #print(ds_day_var)

            # select data only between start_time and end_time
            ds_day_var = ds_day_var.sel(time=slice(start_time, end_time))
            ds_day = ds_day.sel(time=slice(start_time, end_time))

            # count number of timestamps in the data
            num_timestamps = len(ds_day.time.values)
            print(f"Number of timestamps in the data: {num_timestamps}")

            # generate crop centers starting from lat and lon start and evolving to lat/lon end 
            # if start and end centers are both in the frame centered on lat start lon start, then we can generate crops centered on the start location for all timestamps

            # calculate crop centers for the lagrangian crops, based on the start and end location of the case
            crop_centers = define_lagrangian_crop_centers(
                                ds_day_var, x_pixel, y_pixel,
                                lat_start, lon_start, lat_end, lon_end,
                                num_timestamps
                            )
            eulerian_crop_specs = define_eulerian_crop_centers(
                ds_day_var.isel(time=0), x_pixel, y_pixel, expats_domain
            )
            print(f"Eulerian mode will generate {len(eulerian_crop_specs)} views per timestamp.")

            #loop over each timestamp in the daily file selection between start_time and end_time
            #extract timestamp
            timestamps = ds_day_var.time.values
            crop_chunk = []
            chunk_start_index = 0
            eulerian_chunks = {spec['view_id']: [] for spec in eulerian_crop_specs}
            eulerian_chunk_start_indices = {spec['view_id']: 0 for spec in eulerian_crop_specs}
            num_valid_crops = 0
            num_skipped_nan = 0
            num_skipped_range = 0
            num_saved_chunks = 0
            num_valid_eulerian_crops = 0
            num_skipped_eulerian_nan = 0
            num_skipped_eulerian_range = 0
            num_saved_eulerian_chunks = 0

            for ind_time, timestamp in enumerate(timestamps):

                #print(timestamp)
                timestamp_str = str(np.datetime_as_string(timestamp, unit='m'))
                #print(f"Processing timestamp: {timestamp_str}")

                ds_var_full_t = ds_day_var.sel(time=timestamp)
                ds_full_t = ds_day.sel(time=timestamp)
                ds_var_t = ds_var_full_t
                ds_t = ds_full_t
                crop_center_t = crop_centers[ind_time]

                # crop the dataset around the crop center for the timestamp
                ds_var_t = crops_by_center(ds_var_t, x_pixel, y_pixel, crop_center_t)
                ds_t = crops_by_center(ds_t, x_pixel, y_pixel, crop_center_t)

                # Check if all variables in the dataset have any NaN
                is_nan_ds = any([xr.DataArray.isnull(ds_var_t[var]).any() for var in ds_var_t.data_vars])

                # Check if the dataset has values outside the defined range
                is_outside_range = any([((ds_var_t[var] < values_min[i]) | (ds_var_t[var] > values_max[i])).any() for i,var in enumerate(ds_var_t.data_vars)])

                #if there are no Nan, the months is between April and September 
                if not is_nan_ds and not is_outside_range:          
                    num_valid_crops += 1

                    print(f"Processing file: {path_file_bucket} for timestamp: {timestamp_str}")

                    if 'OT 'in var_names:
                        #substitute channl WV_062 with the difference WV_062-IR_108
                        ds_var_t['WV_062'] = ds_var_t['WV_062'] - ds_var_t['IR_108']
                        #the rename the variable to WV_062-IR_108
                        ds_var_t = ds_var_t.rename({'WV_062': 'WV_062-IR_108'})

                    if apply_cma and 'IR_108' in var_names:
                        ds_var_t = add_cloud_mask_and_masked_ir108(ds_t, ds_var_t, values_max, var_names)
                        #print(ds_day_var)

                    if 'time' not in ds_var_t.dims:
                        ds_var_t = ds_var_t.expand_dims(time=[timestamp])

                    crop_chunk.append(ds_var_t)

                    if len(crop_chunk) == chunk_size:
                        save_crop_chunk(
                            crop_chunk, chunk_start_index, outpath_crops, outpath_img,
                            date, file_extension, var_names, var_props, values_min,
                            values_max, save_video, apply_cma, expats_domain,
                            orography=orography, reports_df=case_reports_df
                        )
                        crop_chunk = []
                        chunk_start_index = ind_time + 1
                        num_saved_chunks += 1
                else:
                    if is_nan_ds:
                        num_skipped_nan += 1
                    if is_outside_range:
                        num_skipped_range += 1

                for eulerian_spec in eulerian_crop_specs:
                    eulerian_center = (eulerian_spec['center_lat'], eulerian_spec['center_lon'])
                    ds_var_eulerian_t = crops_by_center(
                        ds_var_full_t, x_pixel, y_pixel, eulerian_center
                    )
                    ds_eulerian_t = crops_by_center(
                        ds_full_t, x_pixel, y_pixel, eulerian_center
                    )

                    is_nan_eulerian = any([
                        xr.DataArray.isnull(ds_var_eulerian_t[var]).any()
                        for var in ds_var_eulerian_t.data_vars
                    ])
                    is_outside_range_eulerian = any([
                        ((ds_var_eulerian_t[var] < values_min[i]) |
                         (ds_var_eulerian_t[var] > values_max[i])).any()
                        for i, var in enumerate(ds_var_eulerian_t.data_vars)
                    ])

                    if is_nan_eulerian or is_outside_range_eulerian:
                        if is_nan_eulerian:
                            num_skipped_eulerian_nan += 1
                        if is_outside_range_eulerian:
                            num_skipped_eulerian_range += 1
                        continue

                    if apply_cma and 'IR_108' in var_names:
                        ds_var_eulerian_t = add_cloud_mask_and_masked_ir108(
                            ds_eulerian_t, ds_var_eulerian_t, values_max, var_names
                        )

                    if 'time' not in ds_var_eulerian_t.dims:
                        ds_var_eulerian_t = ds_var_eulerian_t.expand_dims(time=[timestamp])

                    view_id = eulerian_spec['view_id']
                    eulerian_chunks[view_id].append(ds_var_eulerian_t)
                    num_valid_eulerian_crops += 1

                    if len(eulerian_chunks[view_id]) == chunk_size:
                        save_crop_chunk(
                            eulerian_chunks[view_id], eulerian_chunk_start_indices[view_id],
                            outpath_crops, outpath_img, date, file_extension, var_names,
                            var_props, values_min, values_max, save_video, apply_cma,
                            expats_domain, orography=orography, reports_df=case_reports_df,
                            mode='eulerian', view_spec=eulerian_spec,
                            eulerian_crop_specs=eulerian_crop_specs,
                            video_vars=['IR_108_masked']
                        )
                        eulerian_chunks[view_id] = []
                        eulerian_chunk_start_indices[view_id] = ind_time + 1
                        num_saved_eulerian_chunks += 1

            if crop_chunk:
                save_crop_chunk(
                    crop_chunk, chunk_start_index, outpath_crops, outpath_img,
                    date, file_extension, var_names, var_props, values_min,
                    values_max, save_video, apply_cma, expats_domain,
                    orography=orography, reports_df=case_reports_df
                )
                num_saved_chunks += 1

            for eulerian_spec in eulerian_crop_specs:
                view_id = eulerian_spec['view_id']
                if eulerian_chunks[view_id]:
                    save_crop_chunk(
                        eulerian_chunks[view_id], eulerian_chunk_start_indices[view_id],
                        outpath_crops, outpath_img, date, file_extension, var_names,
                        var_props, values_min, values_max, save_video, apply_cma,
                        expats_domain, orography=orography, reports_df=case_reports_df,
                        mode='eulerian', view_spec=eulerian_spec,
                        eulerian_crop_specs=eulerian_crop_specs,
                        video_vars=['IR_108_masked']
                    )
                    num_saved_eulerian_chunks += 1

            print(
                f"Case {date} summary: {num_valid_crops}/{num_timestamps} valid cropped timestamps, "
                f"{num_skipped_nan} skipped for NaNs, {num_skipped_range} skipped for range, "
                f"{num_saved_chunks} saved chunk(s)."
            )
            print(
                f"Case {date} eulerian summary: {num_valid_eulerian_crops} valid view-timestamps, "
                f"{num_skipped_eulerian_nan} skipped for NaNs, "
                f"{num_skipped_eulerian_range} skipped for range, "
                f"{num_saved_eulerian_chunks} saved chunk(s) across "
                f"{len(eulerian_crop_specs)} view(s)."
            )
                    
if __name__ == "__main__":
    main()






    
