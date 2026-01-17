import os
import xarray as xr
from tqdm import tqdm
from glob import glob
import numpy as np

# === CONFIGURATION ===
input_dir = "/data1/crops/test_case_essl_2021-2025_100x100_ir108_cma/nc/HAIL/1"

# === MAIN SCRIPT ===
nc_files = glob(os.path.join(input_dir, "*.nc"))

print(f"Found {len(nc_files)} .nc files to process in {input_dir}")

for nc_file in tqdm(nc_files, desc="Processing files"):
    file_path = os.path.join(input_dir, nc_file)

    #try:
    ds = xr.open_dataset(file_path)

    # Extract base info from filename
    base_name = os.path.basename(nc_file).replace(".nc", "")
    date_part, lat, lon = base_name.split("_")
    print(f"Processing file: {nc_file} for date: {date_part}, lat: {lat}, lon: {lon}")
    

    # Ensure time dimension exists
    if "time" not in ds.dims:
        print(f"⚠️ Skipping {nc_file}: no time dimension found.")
        ds.close()
        continue

    for t in ds.time:
        # Convert time to string format (YYYY-MM-DDThh:mm)
        ts_str = str(np.datetime_as_string(t.values, unit="m"))
        print(f"  Processing timestep: {ts_str}")
        ds_t = ds.sel(time=t)
        print(ds_t)
          

    #         # Define output filename
    #         out_name = f"{ts_str}_{lat}_{lon}.nc"
    #         out_path = os.path.join(input_dir, out_name)

    #         # Save single-timestep dataset
    #         ds_t.to_netcdf(out_path)
    #         ds_t.close()

    #     ds.close()

    #     # Once done safely, remove the original file
    #     os.remove(file_path)
    #     print(f"✅ Split and deleted original file: {nc_file}")

    # except Exception as e:
    #     print(f"❌ Error processing {nc_file}: {e}")
