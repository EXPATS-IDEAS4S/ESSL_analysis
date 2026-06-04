import pandas as pd
import os
import io
import xarray as xr
import numpy as np
import sys
import boto3
import yaml

sys.path.append('/home/Daniele/codes/ESS_analysis/code/crop_generator/')

from cropping_functions import crops_by_center, apply_cma_mask, convert_crops_to_images
from credentials_buckets import S3_BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL
import aux_func



# Initialize the S3 client
s3 = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY
)

#aux_func.list_all_bucket_objects(s3, S3_BUCKET_NAME)

#define path for retrievg MSG files from bucket
path_dir = f"/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN"
basename = "merged_MSG_CMSAF"

# Define variables properties
path_to_config = "/home/Daniele/codes/ESSL_analysis/code/crop_generator"
with open(path_to_config + "/variables_config.yaml") as f:
    cfg = yaml.safe_load(f)

#Implemented vars are IR_108, WV_062, OT
var_names = ['IR_108'] 

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

print(value_max, values_min)

max_img = 320
min_img = 240

x_pixel = 100 
y_pixel = 100 

#domain = lonmin, lonmax, latmin, latmax = 5, 16, 42, 51.5 #DC domain from the paper

apply_cma = True #if True, the cma variable will be included in the crops
file_extension = 'nc'  # File extension for the dataset files
save_img = True

#output_path =  f'/work/dcorradi/crops/{cloud_prm_str}_{years_str}_{x_pixel}x{y_pixel}_{domain_name}_{cropping_strategy}/'
base_output_path =  "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma_traj"
outpath_crops = f'{base_output_path}'
outpath_img = f'{base_output_path}'
os.makedirs(outpath_crops, exist_ok=True)
os.makedirs(outpath_img, exist_ok=True)

path_csv = f"/home/Daniele/codes/ML_data_generator/test/essl/storm_trajectories_after_merge.csv"

df = pd.read_csv(path_csv, usecols=[
    "merged_storm_id",
    "time",
    "lat",
    "lon",
    "cluster_event_type",
    "source",
])
print(df)


def resolve_storm_type(types):
    s = set(types.dropna())
    if len(s) == 1:
        return list(s)[0]
    return "MIXED"


def resolve_source(sources):
    s = set(sources.dropna())
    if "observed" in s:
        return "observed"
    if "interpolated" in s:
        return "interpolated"
    return "extrapolated"


collapsed = (
    df
    .groupby(["merged_storm_id", "time"])
    .agg(
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        #storm_type=("storm_type", resolve_storm_type),
        cluster_event_type=("cluster_event_type", resolve_storm_type),
        source=("source", resolve_source),
    )
    .reset_index()
    .sort_values(["merged_storm_id", "time"])
)

print(f"Collapsed from {len(df)} → {len(collapsed)} rows")

#check how many unique storm ids are PRECIP, HAIL, MIXED
storm_type_counts = (
    collapsed
    .groupby("merged_storm_id")["cluster_event_type"]
    .first()
    .value_counts()
)
print("Storm type counts after collapsing:")
print(storm_type_counts)



for storm_id, df_storm in collapsed.groupby("merged_storm_id"):

    print(f"\nProcessing merged storm {storm_id} with {len(df_storm)} time steps")

    for _, row in df_storm.iterrows():

        time = row["time"]
        lat = round(row["lat"], 3)
        lon = round(row["lon"], 3)
        storm_type = row["cluster_event_type"]
        source = row["source"]

        crop_center = (lat, lon)

        timestamp_str = pd.to_datetime(time).strftime("%Y-%m-%dT%H-%M")
        #print(timestamp_str)

        #extract year, month, day
        time_pd = pd.to_datetime(time)
        year = time_pd.strftime("%Y")
        month = time_pd.strftime("%m")
        day = time_pd.strftime("%d")

        file = f"{path_dir}/{year}/{month}/{basename}_{year}-{month}-{day}.nc"

        print(
            f"Storm {storm_id} | {timestamp_str} | "
            f"{storm_type} | {source} | ({lat}, {lon})"
        )

        my_obj = aux_func.read_file(s3, file, S3_BUCKET_NAME)
        if my_obj is None:
            continue

        ds_day = xr.open_dataset(io.BytesIO(my_obj))
        ds_day_var = ds_day[var_names]
        #print(ds_day_var.time.values)
        #print(time)
        event_time = pd.to_datetime(row["time"], utc=True)
        event_time_naive = event_time.tz_convert(None)

        try:
            ds_var_t = ds_day_var.sel(time=event_time_naive)
            ds_t = ds_day.sel(time=event_time_naive)

            ds_var_t = crops_by_center(ds_var_t, x_pixel, y_pixel, crop_center)
            ds_t = crops_by_center(ds_t, x_pixel, y_pixel, crop_center)

        except (KeyError, ValueError) as e:
            print(f"Skipping {timestamp_str}: {e}")
            continue

        # quality checks
        is_nan_ds = any(
            xr.DataArray.isnull(ds_var_t[var]).any()
            for var in ds_var_t.data_vars
        )

        is_outside_range = any(
            ((ds_var_t[var] < values_min[i]) |
             (ds_var_t[var] > values_max[i])).any()
            for i, var in enumerate(ds_var_t.data_vars)
        )

        if is_nan_ds or is_outside_range:
            continue

        # optional CMA
        if apply_cma and "cma" in ds_t and "IR_108" in var_names:
            ds_var_t = apply_cma_mask(ds_t, ds_var_t, values_max)

        if "time" not in ds_var_t.dims:
            ds_var_t = ds_var_t.expand_dims("time")

        print(ds_var_t)
        # ---- filename with rich metadata ----
        filename = (
            f"storm{storm_id}"
            f"_{timestamp_str}"
            f"_lat{lat:.2f}_lon{lon:.2f}"
            f"_{storm_type}"
            f"_{source}"   
        )

        print(f"Saving crop: {filename}.{file_extension}")

        save_path = os.path.join(
            outpath_crops,
            file_extension,
            "1",
            f"{filename}.{file_extension}"
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ds_var_t.to_netcdf(save_path)

        print(f"Saved {save_path}")

        if save_img:
            for var, vmax, vmin in zip(var_names, values_max, values_min):
                img_save_path = os.path.join(
                    outpath_img, "images", var
                )
                os.makedirs(img_save_path, exist_ok=True)

                convert_crops_to_images(
                    ds_var_t[var],
                    x_pixel,
                    y_pixel,
                    filename,
                    "png",
                    img_save_path,
                    var_props[var]["cmap"],
                    min_img,
                    max_img,
                    "vmin-vmax",
                    "greyscale",
                    apply_cma,
                )

# 3652962