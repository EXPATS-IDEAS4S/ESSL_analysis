import os
import io
import boto3
import xarray as xr
import sys
import pandas as pd 
import yaml

sys.path.append('/home/Daniele/codes/ESS_analysis/code/crop_generator/')

from cropping_functions import random_crop, apply_cma_mask, convert_crops_to_images
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
# TODO: missing 2024 and 2025 files in MSG bucket

#define path for retrievg MSG files from bucket
path_dir = f"/data/sat/msg/ml_train_crops/IR_108-WV_062-CMA_FULL_EXPATS_DOMAIN"
basename = "merged_MSG_CMSAF"

# Define variables properties
path_to_config = "/home/dcorradi/Documents/Codes/ESSL_analysis/code/crop_generator"
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

x_pixel = 128 
y_pixel = 128 

#domain = lonmin, lonmax, latmin, latmax = 5, 16, 42, 51.5 #DC domain from the paper

apply_cma = True #if True, the cma variable will be included in the crops
file_extension = 'nc'  # File extension for the dataset files
save_img = True

#output_path =  f'/work/dcorradi/crops/{cloud_prm_str}_{years_str}_{x_pixel}x{y_pixel}_{domain_name}_{cropping_strategy}/'
outpath_crops = f'/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops_control/{x_pixel}x{y_pixel}/{file_extension}'
outpath_img = f'/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/crops_control/{x_pixel}x{y_pixel}/images'
os.makedirs(outpath_crops, exist_ok=True)
os.makedirs(outpath_img, exist_ok=True)

events_name = ['PRECIP', 'HAIL']
path_csv = "/work/dcorradi/ESSL/conference_analysis_2025/grouped_output/"

years_range = range(2021, 2026) # 2021 to 2025
months_range = range(4, 10)  # April to September

n_tot_events = 0
unique_days_all_events = pd.DatetimeIndex([])
for event in events_name:
    print(f"Processing event: {event}")
    # Open the dataset with the list of events

    csv_file_event = f"{event}_summary.csv"

    #open the csv files
    df_event = pd.read_csv(path_csv+csv_file_event)
    #print(df_event.columns)

    n_events = len(df_event)

    #check how many events
    n_tot_events += n_events

    print(f"Total number of events: {n_events}")

    #list of unique days
    unique_days = df_event['day_id'].nunique()
    print(f"{event}: {n_events} events across {unique_days} unique days")

    unique_days_all_events = unique_days_all_events.union(pd.to_datetime(df_event['day_id'].unique()))

print(f"Total unique days across all events: {len(unique_days_all_events)}")

#generate randomly n_events days that do not overlap with the unique days in the defined range
#generate a list fro each year in the range and then merge them (to exclude unwanted months)
all_days = pd.DatetimeIndex([])
for year in years_range:
    if year == years_range[0]:
        all_days = pd.date_range(start=f'{year}-{months_range[0]}-01', end=f'{year}-{months_range[-1]}-30', freq='D')
    else:
        days_year = pd.date_range(start=f'{year}-{months_range[0]}-01', end=f'{year}-{months_range[-1]}-30', freq='D')
        all_days = all_days.union(days_year)

non_event_days = all_days.difference(unique_days_all_events)
print(non_event_days)

random_control_days = non_event_days.to_series().sample(n=n_tot_events, random_state=42, replace=True).sort_values().reset_index(drop=True)
print(random_control_days)

#loop through each event day
for day in random_control_days:
    day_id = day.strftime('%Y-%m-%d')
    print(f"Processing control day: {day_id}")
    
    #get year, month, day from day_id
    year = day_id.split('-')[0]
    month = day_id.split('-')[1]
    day = day_id.split('-')[2]
    #print(f"Year: {year}, Month: {month}, Day: {day}")

    file = f"{path_dir}/{year}/{month}/{basename}_{year}-{month}-{day}.nc"
    #print(file)

    #Read file from the bucket
    my_obj = aux_func.read_file(s3, file, S3_BUCKET_NAME)
    if my_obj is not None:
        ds_day = xr.open_dataset(io.BytesIO(my_obj))
        #print(ds_day)
        
        #select only variable of interest
        ds_day_var = ds_day[var_names]
        #print(ds_day_var)

        #select only data within certain domain
        try:
            ds_day_var = random_crop(ds_day_var, x_pixel, y_pixel)
            ds_day = random_crop(ds_day, x_pixel, y_pixel)
        except ValueError as e:
            print(f"Skipping file '{file}' due to error: {e}")
            continue  # Skip this file and move to the next
        
        # Check if all variables in the dataset have any NaN
        is_nan_ds = any([xr.DataArray.isnull(ds_day_var[var]).any() for var in ds_day_var.data_vars])

        # Check if the dataset has values outside the defined range
        is_outside_range = any([((ds_day_var[var] < values_min[i]) | (ds_day_var[var] > values_max[i])).any() for i,var in enumerate(ds_day_var.data_vars)])

        #if there are no Nan, the months is between April and September 
        if not is_nan_ds and not is_outside_range:          
            print(f"Processing file: {file} for timestamp: {day_id}")
            # saving cropped images using a filename based on day_id and cluster_center
            filename_to_save = str(day_id)+f"_random"
            #print(filename_to_save)

            if 'OT 'in var_names:
                #print(f"Applying OT to {filename_to_save}")
                #substitute channl WV_062 with the difference WV_062-IR_108
                ds_day_var['WV_062'] = ds_day_var['WV_062'] - ds_day_var['IR_108']
                #the rename the variable to WV_062-IR_108
                ds_day_var = ds_day_var.rename({'WV_062': 'WV_062-IR_108'})

            if apply_cma and 'cma' in ds_day and 'IR_108' in var_names:
                #apply value filds depending if OT is True or False
                ds_day_var = apply_cma_mask(ds_day, ds_day_var, values_max)
                #print(ds_day_var)
                

            # Save the processed dataset
            save_path = os.path.join(outpath_crops, '1', f"{filename_to_save}.{file_extension}")
            os.makedirs(os.path.join(outpath_crops, '1'), exist_ok=True)
            ds_day_var.to_netcdf(save_path)
            print(f"Saved cropped dataset to {save_path}")

            # If save_img is True, save images of the variables
            if save_img:
                for var, vmax, vmin in zip(var_names, values_max, values_min):
                    img_save_path = os.path.join(outpath_img, var)
                    os.makedirs(img_save_path, exist_ok=True)
                    
                    ds_sel = ds_day_var[var]
                    cmap = var_props[var]['cmap']
                    #plot and save the image
                    convert_crops_to_images(ds_sel, x_pixel, y_pixel, filename_to_save, 'png', img_save_path, cmap, vmin, vmax, 'vmin-vmax', 'greyscale', apply_cma)
                    print(f"Saved image to {img_save_path}")

                    
#nohup 1889477