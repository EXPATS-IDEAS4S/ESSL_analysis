import os
import xarray as xr
import numpy as np

# === CONFIGURATION ===
input_dir = "/data1/crops/test_case_essl_2021-2025_100x100_ir108_cma/PRECIP/nc/1"

print(f"🔍 Checking .nc files in: {input_dir}")

nc_files = [f for f in os.listdir(input_dir) if f.endswith(".nc")]
modified = 0

for nc_file in nc_files:
    file_path = os.path.join(input_dir, nc_file)

    try:
        ds = xr.open_dataset(file_path, engine="h5netcdf")

        if "time" not in ds.dims:
            # Check if there is a coordinate variable called 'time'
            if "time" in ds.coords:
                time_values = ds["time"].values
                # Ensure iterable for expand_dims
                if np.isscalar(time_values):
                    time_values = [time_values]
                ds = ds.expand_dims({"time": time_values})
                ds.to_netcdf(file_path, engine="h5netcdf")
                modified += 1
                print(f"✅ Added 'time' dimension to {nc_file} from coordinate")
            else:
                print(f"⚠️ {nc_file} has no 'time' dimension or coordinate, skipped")

        else:
            print(f"✅ {nc_file} already has 'time' dimension")

        ds.close()

    except Exception as e:
        print(f"❌ {nc_file}: error -> {e}")

print(f"\n✅ Done! Modified {modified} files.")

#1678432