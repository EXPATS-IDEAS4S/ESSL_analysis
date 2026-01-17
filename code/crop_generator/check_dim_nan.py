import os
import re
import shutil
import xarray as xr

# === CONFIGURATION ===
input_dir = "/data1/crops/test_case_essl_2021-2025_100x100_ir108_cma/PRECIP/nc/1"
out_dir = "/home/Daniele/codes/ESSL_analysis/code/crop_generator"
output_txt = os.path.join(out_dir, "invalid_nc_files.txt")
nan_dir = "/data1/crops/test_case_essl_2021-2025_100x100_ir108_cma/PRECIP/nc_nan"

# Ensure output directories exist
os.makedirs(out_dir, exist_ok=True)
os.makedirs(nan_dir, exist_ok=True)

# Expected grid size
expected_lat = 100
expected_lon = 100

invalid_files = []
nan_days = set()

print(f"🔍 Checking .nc files in: {input_dir}")

nc_files = [f for f in os.listdir(input_dir) if f.endswith(".nc")]
datetime_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")

for nc_file in nc_files:
    file_path = os.path.join(input_dir, nc_file)
    try:
        ds = xr.open_dataset(file_path)
        print(ds)
        # data = ds['IR_108'].values
        # print(data.shape)
        # import torch
        # import numpy as np
        # #generate random 100x100 crop
        # random_crop = data
        # channels = []
        # channels.append(data)
        # #channels.append(random_crop)
        # tensor = torch.tensor(np.stack(channels, axis=0), dtype=torch.float32)
        # print(tensor.shape)
        # exit()
        # channels = []
        # for var in ds.data_vars:
        #     if var not in ds:
        #         print(f"Variable {var} not found in {file_path}, skipping.")
        #         continue
        #     data = ds[var].isel(time=0).values
        #     channels.append(data)
        # print(channels)
        # exit()

        # --- Basic structure checks ---
        if not ds.data_vars:
            invalid_files.append(f"{nc_file}: no data variables")
            ds.close()
            continue

        # Identify coordinate names
        lat_name = "lat" if "lat" in ds.dims else "latitude"
        lon_name = "lon" if "lon" in ds.dims else "longitude"

        lat_dim = ds.dims.get(lat_name)
        lon_dim = ds.dims.get(lon_name)

        # === Check grid dimensions ===
        if lat_dim != expected_lat or lon_dim != expected_lon:
            invalid_files.append(f"{nc_file}: lat={lat_dim}, lon={lon_dim}")
            ds.close()
            continue

        # === Check for time dimension ===
        if "time" not in ds.dims:
            invalid_files.append(f"{nc_file}: missing 'time' dimension")
            ds.close()
            continue

        # === Check variables for consistency ===
        invalid_vars = []
        for var_name, var_data in ds.data_vars.items():
            # Expect (time, lat, lon) or (lat, lon)
            dims = var_data.dims
            if dims not in [("time", lat_name, lon_name), (lat_name, lon_name)]:
                invalid_vars.append(f"{var_name} dims={dims}")
                continue

            # Check for NaNs or empty arrays
            if var_data.size == 0 or var_data.isnull().all():
                invalid_vars.append(f"{var_name} is empty or all NaN")

        if invalid_vars:
            invalid_files.append(f"{nc_file}: invalid variables -> {', '.join(invalid_vars)}")
            ds.close()
            continue

        # === Global NaN check ===
        has_nan = ds.to_array().isnull().any().item()
        ds.close()

        if has_nan:
            match = datetime_pattern.search(nc_file)
            if match:
                datetime_str = match.group(0)
                nan_days.add(datetime_str)

    except Exception as e:
        invalid_files.append(f"{nc_file}: ERROR ({e})")

# === Move files with NaN to nan_dir ===
moved_files = []
if nan_days:
    for nc_file in nc_files:
        match = datetime_pattern.search(nc_file)
        if match and match.group(0) in nan_days:
            src = os.path.join(input_dir, nc_file)
            dst = os.path.join(nan_dir, nc_file)
            shutil.move(src, dst)
            moved_files.append(nc_file)

# === SAVE RESULTS ===
with open(output_txt, "w") as f:
    if invalid_files:
        f.write("⚠️ Files with invalid structure or variables:\n")
        f.write("\n".join(invalid_files))
        f.write("\n\n")
    if moved_files:
        f.write("🧹 Files moved due to NaN values (by day):\n")
        f.write("\n".join(moved_files))
        f.write("\n\n")
    if not invalid_files and not moved_files:
        f.write("✅ All files passed all checks.\n")

print("\n✅ Pre-check complete.")
print(f"⚠️ Invalid files: {len(invalid_files)}")
print(f"🧹 Days with NaNs: {len(nan_days)}")
print(f"📦 Files moved to '{nan_dir}': {len(moved_files)}")
print(f"🧾 Results saved to: {output_txt}")
