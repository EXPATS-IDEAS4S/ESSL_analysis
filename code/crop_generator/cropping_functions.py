import xarray as xr
from random import randrange
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
from scipy.ndimage import binary_closing


def crops_nc_fixed(ds_image, x_pixel, y_pixel, crop_positions, filename, out_path, file_type = 'nc'):
    """
    Generates fixed crops from the input dataset and saves them in NetCDF and TIFF formats.

    This function takes an input dataset, generates fixed crops based on specified positions,
    and saves the crops as NetCDF files to preserve the original values and coordinates.

    :param ds_image: xarray.Dataset or xarray.DataArray
        The input dataset containing the image data along with latitude and longitude coordinates.
    :param x_pixel: int
        The width of the crop in pixels.
    :param y_pixel: int
        The height of the crop in pixels.
    :param crop_positions: list of tuples
        List of tuples, where each tuple contains the (lat, lon) of the upper-left corner of a crop.
    :param filename: str
        The base filename for saving the cropped images.
    :param out_path: str
        The output directory where the cropped images will be saved.
    
    :return: None
    """
    for i, (lat_ul, lon_ul) in enumerate(crop_positions):
        #print(f"lat: {lat_ul}, lon: {lon_ul}")
        # Find indices of the upper-left lat/lon in the dataset
        x1 = int((lon_ul - ds_image.lon.values.min()) / (ds_image.lon.values[1] - ds_image.lon.values[0]))
        y1 = int((lat_ul - ds_image.lat.values.min()) / (ds_image.lat.values[1] - ds_image.lat.values[0]))

        # Calculate crop boundaries
        latmax = ds_image.lat.values[y1]
        latmin = ds_image.lat.values[y1 - y_pixel + 1]
        lonmin = ds_image.lon.values[x1]
        lonmax = ds_image.lon.values[x1 + x_pixel - 1]
        #print([lonmin, lonmax, latmin, latmax])

        # Crop the dataset
        ds_crop = filter_by_domain(ds_image, [lonmin, lonmax, latmin, latmax])

        #check if the crop contains any Nan
        isnan_ds = xr.DataArray.isnull(xr.DataArray.sum(ds_crop,skipna=False))

        if isnan_ds==False:
            # Save the crop as NetCDF
            if file_type == 'nc':
                nc_path = f"{out_path}/nc/{filename}_{i}.nc"
                ds_crop.to_netcdf(nc_path, engine="h5netcdf")
                print(f"{nc_path} saved")
            elif file_type == 'npy':
                npy_path = f"{out_path}/{filename}_{i}.npy"
                #print(ds_crop)
                #print(ds_crop.to_array()[0,:,:,:].values)	
                array_to_save = ds_crop.to_array()[0,:,:,:].values  
                #save if do not cantain any Nan
                if not np.isnan(array_to_save).any(): 
                    np.save(npy_path, array_to_save)
                    print(f"{npy_path} saved")
            else:
                raise ValueError(f"Invalid file type: '{file_type}'")




def random_crop(ds_image, x_pixel, y_pixel, seed=None):
    """
    Extracts a random fixed-size crop (x_pixel × y_pixel) from the dataset.

    Parameters
    ----------
    ds_image : xarray.Dataset or xarray.DataArray
        The input dataset containing 'lat' and 'lon' coordinates.
    x_pixel : int
        Width of the crop in pixels.
    y_pixel : int
        Height of the crop in pixels.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ds_crop : xarray.Dataset or xarray.DataArray
        Randomly cropped subset of the input dataset.
    """

    # Ensure lat/lon dimensions exist
    if not {"lat", "lon"} <= set(ds_image.dims):
        raise ValueError("Dataset must have 'lat' and 'lon' dimensions")

    n_x = len(ds_image.lon)
    n_y = len(ds_image.lat)

    # Ensure crop size fits inside the dataset
    if x_pixel > n_x or y_pixel > n_y:
        raise ValueError("Requested crop size exceeds dataset dimensions")

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    # Randomly select top-left corner ensuring crop fits within bounds
    x_start = rng.integers(0, n_x - x_pixel + 1)
    y_start = rng.integers(0, n_y - y_pixel + 1)

    # Compute end indices (exclusive)
    x_end = x_start + x_pixel
    y_end = y_start + y_pixel

    # Crop by index
    ds_crop = ds_image.isel(
        lon=slice(x_start, x_end),
        lat=slice(y_start, y_end)
    )

    return ds_crop



def crops_by_center(ds_image, x_pixel, y_pixel, crop_center):
    """
    Extracts a fixed-size crop centered at a given lat/lon position.

    Returns exactly x_pixel × y_pixel pixels.
    """
    lat_c, lon_c = crop_center

    # Find nearest pixel to the requested center
    x_center = np.abs(ds_image.lon.values - lon_c).argmin()
    y_center = np.abs(ds_image.lat.values - lat_c).argmin()

    # Half-widths in pixels
    half_x = x_pixel // 2
    half_y = y_pixel // 2

    # Adjust ranges to ensure correct crop size (end-exclusive)
    if x_pixel % 2 == 0:
        x_start = x_center - half_x
        x_end = x_center + half_x      # end-exclusive
    else:
        x_start = x_center - half_x
        x_end = x_center + half_x + 1  # center included, odd size

    if y_pixel % 2 == 0:
        y_start = y_center - half_y
        y_end = y_center + half_y
    else:
        y_start = y_center - half_y
        y_end = y_center + half_y + 1

    # Clip to dataset bounds
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(len(ds_image.lon), x_end)
    y_end = min(len(ds_image.lat), y_end)

    # Index-based selection (avoids inclusive slicing ambiguity)
    ds_crop = ds_image.isel(
        lon=slice(x_start, x_end),
        lat=slice(y_start, y_end)
    )

    return ds_crop





def crops_nc_random(ds_image, x_pixel, y_pixel, n_sample, filename, out_path, file_type = 'nc'):
    """
    Generates multiple random crops from the input dataset, saves them in NetCDF and TIFF formats.

    This function takes an input dataset, generates random crops of specified size, and saves
    the crops as NetCDF files to preserve the original values and coordinates. Additionally,
    it saves the cropped images as TIFF files.

    :param ds_image: xarray.Dataset or xarray.DataArray
        The input dataset containing the image data along with latitude and longitude coordinates.
    :param x_pixel: int
        The width of the crop in pixels.
    :param y_pixel: int
        The height of the crop in pixels.
    :param n_sample: int
        The number of random crops to generate.
    :param filename: str
        The base filename for saving the cropped images.
    :param out_path: str
        The output directory where the cropped images will be saved.
    
    :return: None
    """

    #get array size from dataset ds_image 
    x = len(ds_image.lon.values)
    y = len(ds_image.lat.values)
    #print(x,y)
       
    for i in range(n_sample):

        x1 = randrange(0, x - x_pixel)
        y1 = randrange(0, y - y_pixel)
        #print(x1,y1)

        #find lat lon boundaries of the random crop
        latmin = ds_image.lat.values[y1]
        latmax = ds_image.lat.values[y1+y_pixel-1]
        lonmin = ds_image.lon.values[x1]
        lonmax = ds_image.lon.values[x1+x_pixel-1]
        #print([lonmin, lonmax, latmin, latmax])

        #crop the dataset besed on the random x and y (the upper left point of the crop)
        ds_crop = filter_by_domain(ds_image,[lonmin, lonmax, latmin, latmax])

        #check if the crop contains any Nan
        isnan_ds = xr.DataArray.isnull(xr.DataArray.sum(ds_crop,skipna=False))
        #if not ds_crop.to_array().isnull().any().item():

        if isnan_ds==False:
            #save the crops in nc format to keep actual values and lat/lon coordinates
            filepath = out_path+'/'+filename+"_"+str(i)+'.'+file_type
            #print(filepath)

            if file_type == 'nc':
                encoding = {
                                var: {
                                    'zlib': True,
                                    'complevel': 9,
                                    'dtype': ds_crop[var].dtype.name  # preserve original dtype (e.g., 'float32', 'int16')
                                } for var in ds_crop.data_vars
                            }
                ds_crop.to_netcdf(filepath, encoding=encoding, engine="h5netcdf")
            elif file_type == 'npy':
                #print(ds_crop.to_array().values.shape)
                array_to_save = ds_crop.to_array()[0,:,:].values
                np.save(filepath, array_to_save)
            else:
                raise ValueError(f"Invalid file type: '{file_type}'")

            print(out_path+'/'+filename+"_"+str(i), 'saved')

        #close the dataset to free resources
        ds_crop.close()


def filter_by_domain(ds, domain):
    """
    Filters the input dataset based on a specified geographical domain.

    This function applies a mask to the input dataset to include only the data
    points within the specified latitude and longitude boundaries. The masked 
    dataset is then returned.

    :param ds: xarray.Dataset or xarray.DataArray
        The input dataset containing the data to be filtered.
    :param domain: list or tuple of float
        A list or tuple specifying the domain boundaries in the format 
        [lonmin, lonmax, latmin, latmax], where:
        - lonmin: Minimum longitude
        - lonmax: Maximum longitude
        - latmin: Minimum latitude
        - latmax: Maximum latitude
    
    :return: xarray.Dataset or xarray.DataArray
        The dataset filtered by the specified geographical domain.
    """
    lonmin, lonmax, latmin, latmax = domain
    #Find the mask
    mask = (ds['lat'] >= latmin) & (ds['lat'] <= latmax) & (ds['lon'] >= lonmin) & (ds['lon'] <= lonmax)
    
    # Apply the mask to the dataset
    ds_masked = ds.where(mask, drop=True)

    return ds_masked


def filter_by_time(ds, times):
    """
    Filters the input dataset based on a list of specified times.

    :param ds: xarray.Dataset or xarray.DataArray
        The input dataset containing the data to be filtered.
    :param times: datetime-like or list/array of datetimes
        One or more times to filter the dataset by.
    
    :return: xarray.Dataset or xarray.DataArray
        The dataset filtered by the specified times.
    """
    mask = ds['time'].isin(times)   # works with single or multiple timestamps
    ds_masked = ds.where(mask, drop=True)
    return ds_masked



def compute_global_min_max(file_paths, variable):
    """
    Compute the global minimum and maximum values across all NetCDF datasets.

    :param file_paths: List of paths to NetCDF files.
    :param variable: The variable to analyze.
    :return: (global_min, global_max) values.
    """
    global_min = float('inf')
    global_max = float('-inf')

    for path in file_paths:
        ds = xr.open_dataset(path)
        data = ds[variable].values
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        
        # Update global min and max
        global_min = min(global_min, data_min)
        global_max = max(global_max, data_max)
        
        ds.close()

    return global_min, global_max



def create_fig(image, pixel_size, cmap, vmin=None, vmax=None, flip=True):
    """
    Creates and returns a matplotlib figure with the given pixel size from an image.

    This function generates a matplotlib figure using the specified image and pixel size. 
    The figure is created with `dpi=1` to ensure the pixel size directly corresponds 
    to the figure size.

    :param image: The image to be plotted, compatible with matplotlib's imshow function.
    :param pixel_size: A tuple representing the pixel size (width, height) to be used as the figure size.
    :param vmin: The minimum data value that corresponds to the colormap's lower limit.
    :param vmax: The maximum data value that corresponds to the colormap's upper limit.
    :param flip: If True, the image will be flipped upside down before being plotted.
    :return: The created matplotlib figure.
    """
    if flip:
        image = np.flipud(image)
    fig, ax = plt.subplots(figsize=pixel_size, dpi=1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis(False)
    plt.close(fig)
    return fig



def convert_crops_to_images(ds_image, x_pixel, y_pixel, filename, format, out_path, cmap, vmin, vmax, norm_type, color_mode, apply_cma):
    """
    Generates a cropped image from the input dataset and saves it in the specified format.

    This function takes an input dataset and generates a cropped image based on the specified 
    pixel dimensions (x_pixel, y_pixel). The cropped image is saved using the specified format 
    (e.g., TIFF, PNG) and file path. The function uses the provided colormap and value range 
    (vmin, vmax) to enhance the image visualization. It also allows saving images in RGB or greyscale.

    Parameters:
    -----------
    ds_image : xarray.Dataset or xarray.DataArray
        The input dataset or data array containing the image data, typically including 
        associated latitude and longitude coordinates.
    
    x_pixel : int
        The width of the crop in pixels.
    
    y_pixel : int
        The height of the crop in pixels.
    
    filename : str
        The base filename used to save the cropped image.
    
    format : str
        The format in which the cropped image will be saved (e.g., 'tiff', 'png').
    
    out_path : str
        The directory path where the cropped image will be saved.
    
    cmap : str
        The colormap to use for visualizing the image.
    
    vmin : float
        The minimum value for color scaling in the image visualization.
    
    vmax : float
        The maximum value for color scaling in the image visualization.
    
    color_mode : str, optional (default='RGB')
        The color mode for saving the image, either 'RGB' or 'greyscale'.

    Returns:
    --------
    None
        The function does not return any value. It saves the cropped image to the specified file path.
    """
    # Define output directory
    if apply_cma:
        out_dir = f'{out_path}/{format}_{norm_type}_{color_mode}_CMA'
    else:
        out_dir = f'{out_path}/{format}_{norm_type}_{color_mode}'

    os.makedirs(out_dir, exist_ok=True)

    times = ds_image['time'].values

    for t in times:
        #select the timestamp
        ds_time = ds_image.sel(time=t)
        #save the images
        fig = create_fig(ds_time.values.squeeze(),[x_pixel,y_pixel], cmap, vmin, vmax)

        #save using t with yyymmddhhmm format
        t = np.datetime_as_string(t, unit='m').replace('-', '').replace(':', '')
        crop_filepath = f'{out_dir}/{filename}_{t}.{format}'

        fig.savefig(crop_filepath, dpi=1)

        print(f'{crop_filepath} is saved')

        # Open the saved image for conversion
        image = PIL.Image.open(crop_filepath)

        # Convert based on the specified color mode
        if color_mode == 'RGB':
            converted_image = image.convert('RGB')
            converted_image.save(crop_filepath)
            print(f'{crop_filepath} converted to RGB')
        elif color_mode == 'greyscale':
            converted_image = image.convert('L')  # 'L' mode is greyscale in PIL
            converted_image.save(crop_filepath)
            print(f'{crop_filepath} converted to greyscale')
        else: 
            print('color mode not recognized!')

        # Close the image to free resources
        converted_image.close()
        image.close()



def convert_crops_to_images_1band(ds_image, x_pixel, y_pixel, filename, format, out_path, vmin=None, vmax=None, norm_type=None):
    """
    Generates a cropped grayscale image from the input dataset and saves it in the specified format.

    Parameters:
    -----------
    ds_image : xarray.Dataset or xarray.DataArray
        The input dataset or data array containing the image data.
    
    x_pixel : int
        The width of the crop in pixels.
    
    y_pixel : int
        The height of the crop in pixels.
    
    filename : str
        The base filename used to save the cropped image.
    
    format : str
        The format in which the cropped image will be saved (e.g., 'tiff', 'png').
    
    out_path : str
        The directory path where the cropped image will be saved.
    
    vmin : float, optional
        The minimum value for color scaling in the image visualization.
    
    vmax : float, optional
        The maximum value for color scaling in the image visualization.
    
    norm_type : str, optional
        The normalization type to be used for saving. Default is None.
    
    Returns:
    --------
    None
        The function saves the cropped image to the specified file path.
    """
    
    # Normalize the data if vmin and vmax are provided
    if vmin is not None and vmax is not None:
        data = np.clip(ds_image.values, vmin, vmax)
        data = (data - vmin) / (vmax - vmin) * 255.0
    else:
        data = ds_image.values
    
    # Convert the data to uint8 type
    data = data.astype(np.uint8)

    # Create a PIL image from the numpy array
    image = PIL.Image.fromarray(data, mode='L')  # 'L' mode is for grayscale

    # Define output directory
    out_dir = f'{out_path}{format}_{norm_type}'

    # Check if the directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir) 

    # Define the complete file path
    crop_filepath = f'{out_dir}/{filename}_{norm_type}.{format}'

    # Save the image
    image.save(crop_filepath)

    print(f'{crop_filepath} is saved')


def apply_cma_mask(ds_day, ds_day_var, value_max):#, only_108=True):
    """
    Applies binary closing to each time slice of the 'cma' field in ds_day,
    and updates 'ir_108' in ds_day_var accordingly at each timestamp.

    Parameters:
    - ds_day: xarray Dataset with dimensions (T, lat, lon) containing 'cma'.
    - ds_day_var: xarray Dataset with the same time dimension, containing 'ir_108'.
    - cloud_prm: list of strings indicating which cloud parameters are present.
    - value_max: list of value to assign where closed CMA mask is 0.

    Returns:
    - Updated ds_day_var with masked 'ir_108' for each time step.
    """
    
    structure = np.ones((3, 3), dtype=np.uint8)

    for t in ds_day['time']:
        cma_slice = ds_day['cma'].sel(time=t).values
        closed_cma = binary_closing(cma_slice, structure=structure)

        # if only_108:
        #     # Only apply mask to 'IR_108'
        #     ir_108_slice = ds_day_var['IR_108'].sel(time=t)
        #     masked_ir_108 = ir_108_slice.where(closed_cma == 1, value_max[0])
        #     ds_day_var['IR_108'].loc[dict(time=t)] = masked_ir_108
        # else:
        # Apply mask to all variables in ds_day_var, apply for each variable the corresponding value_max
        for var, val_mask in zip(ds_day_var.data_vars, value_max):
            var_slice = ds_day_var[var].sel(time=t)
            masked_var = var_slice.where(closed_cma == 1, val_mask)
            ds_day_var[var].loc[dict(time=t)] = masked_var
        
    return ds_day_var