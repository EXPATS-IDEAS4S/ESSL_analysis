import os
import xarray as xr
import numpy as np
import torch

# Directory containing .nc files
input_dir = "/data1/crops/test_case_essl_2021-2025_100x100_ir108_cma/PRECIP/nc/1"

print(f"🔍 Checking stacking for files in: {input_dir}")

for nc_file in sorted(os.listdir(input_dir)):
    if not nc_file.endswith(".nc"):
        continue
    file_path = os.path.join(input_dir, nc_file)

    try:
        ds = xr.open_dataset(file_path)

        channels = []
        for var in ds.data_vars:
            data = ds[var]
            if "time" in data.dims:
                data = data.isel(time=0)
            arr = data.values

            if arr.ndim == 2 and not np.isnan(arr).all():
                channels.append(arr)

        ds.close()

        if not channels:
            print(f"❌ {nc_file}: no valid 2D data to stack")
            continue

        tensor = torch.tensor(np.stack(channels, axis=0), dtype=torch.float32)
        print(f"✅ {nc_file}: stacked shape {tuple(tensor.shape)}")

    except Exception as e:
        print(f"❌ {nc_file}: error -> {e}")
