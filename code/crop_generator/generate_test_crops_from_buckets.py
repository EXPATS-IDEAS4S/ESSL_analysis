import os
import io
import boto3
import xarray as xr
import sys
import pandas as pd 
import yaml
import numpy as np

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

x_pixel = 100 
y_pixel = 100 

#domain = lonmin, lonmax, latmin, latmax = 5, 16, 42, 51.5 #DC domain from the paper

apply_cma = True #if True, the cma variable will be included in the crops
file_extension = 'nc'  # File extension for the dataset files
save_img = True

#output_path =  f'/work/dcorradi/crops/{cloud_prm_str}_{years_str}_{x_pixel}x{y_pixel}_{domain_name}_{cropping_strategy}/'
base_output_path =  "/data1/crops/test_case_essl_14-15-16-18-19-20-22-23-24_100x100_ir108_cma"
outpath_crops = f'{base_output_path}'
outpath_img = f'{base_output_path}'
os.makedirs(outpath_crops, exist_ok=True)
os.makedirs(outpath_img, exist_ok=True)

events_name = ['PRECIP', 'HAIL']
path_csv = f"{base_output_path}/"

for event in events_name:
    print(f"Processing event: {event}")
    # Open the dataset with the list of events

    csv_file_event = f"{event}_summary.csv"

    #open the csv files
    df_event = pd.read_csv(path_csv+csv_file_event)
    #print(df_event.columns)

    #Check the event day summary
    aux_func._print_event_day_summary(df_event, event)
    
    #loop through each event day
    for index, row in df_event.iterrows():
        #print(row)
        #from row get day_id, cluster_lat and cluster_lon
        day_id = row['day_id']
        #round to second decimal place
        cluster_lat = round(row['cluster_lat'], 2)
        cluster_lon = round(row['cluster_lon'], 2)
        crop_center = (cluster_lat, cluster_lon)
        print(f"Processing event on day {day_id} at location ({cluster_lat}, {cluster_lon})")
        
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

            #loop over each timestamp in the daily file
            #extract timestamp
            timestamps = ds_day_var.time.values
           
            for timestamp in timestamps:
                #print(timestamp)
                timestamp_str = str(np.datetime_as_string(timestamp, unit='m'))
                #print(f"Processing timestamp: {timestamp_str}")
                
                ds_var_t = ds_day_var.sel(time=timestamp)
                ds_t = ds_day.sel(time=timestamp)

                #select only data within certain domain
                try:
                    ds_var_t = crops_by_center(ds_var_t, x_pixel, y_pixel, crop_center)
                    ds_t = crops_by_center(ds_t, x_pixel, y_pixel, crop_center)
                except ValueError as e:
                    print(f"Skipping file '{file}' due to error: {e}")
                    continue  # Skip this file and move to the next
            
                #print lat lon dimensions to check
                #print(ds_var_t.dims)
                #print(ds_t.dims)

                # Check if all variables in the dataset have any NaN
                is_nan_ds = any([xr.DataArray.isnull(ds_var_t[var]).any() for var in ds_var_t.data_vars])

                # Check if the dataset has values outside the defined range
                is_outside_range = any([((ds_var_t[var] < values_min[i]) | (ds_var_t[var] > values_max[i])).any() for i,var in enumerate(ds_var_t.data_vars)])

                #if there are no Nan, the months is between April and September 
                if not is_nan_ds and not is_outside_range:          
                    print(f"Processing file: {file} for timestamp: {timestamp_str}")
                    # saving cropped images using a filename based on day_id and cluster_center
                    filename_to_save = timestamp_str+'_'+str(cluster_lat)+'_'+str(cluster_lon)

                    if 'OT 'in var_names:
                        #print(f"Applying OT to {filename_to_save}")
                        #substitute channl WV_062 with the difference WV_062-IR_108
                        ds_var_t['WV_062'] = ds_var_t['WV_062'] - ds_var_t['IR_108']
                        #the rename the variable to WV_062-IR_108
                        ds_var_t = ds_var_t.rename({'WV_062': 'WV_062-IR_108'})

                    if apply_cma and 'cma' in ds_t and 'IR_108' in var_names:
                        #apply value filds depending if OT is True or False
                        ds_var_t = apply_cma_mask(ds_t, ds_var_t, values_max)
                        #print(ds_day_var)
                        

                    # Save the processed dataset
                    save_path = os.path.join(outpath_crops, event, file_extension, '1', f"{filename_to_save}.{file_extension}")
                    os.makedirs(os.path.join(outpath_crops, event, file_extension, '1'), exist_ok=True)
                    #add time dimension back if needed
                    if 'time' not in ds_var_t.dims:
                        ds_var_t = ds_var_t.expand_dims("time")
                        #print(ds_var_t)
                    
                    ds_var_t.to_netcdf(save_path)
                    print(f"Saved cropped dataset to {save_path}")

                    # If save_img is True, save images of the variables
                    if save_img:
                        for var, vmax, vmin in zip(var_names, values_max, values_min):
                            img_save_path = os.path.join(outpath_img, event, 'images', var)
                            os.makedirs(img_save_path, exist_ok=True)

                            ds_sel = ds_var_t[var]
                            cmap = var_props[var]['cmap']
                            #plot and save the image
                            convert_crops_to_images(ds_sel, x_pixel, y_pixel, filename_to_save, 'png', img_save_path, cmap, vmin, vmax, 'vmin-vmax', 'greyscale', apply_cma)
                            print(f"Saved image to {img_save_path}")
                       
#nohup 86693