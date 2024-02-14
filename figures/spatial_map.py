"""
function to plot spatial map of events

"""

import pandas as pd

from readers.csv import read_csv, read_csv_all
from readers.filenames_paths import path_file, path_figs, raster_filename, filename_climatology
from mpl_style import colors_list, CMAP
from domain_info import domain_red, domain_grey, domain_dfg, det_domain
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import rasterio
import numpy as np
import xarray as xr
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from matplotlib.colors import Normalize

from mpl_toolkits.basemap import Basemap


def main():
    
    # read csv files
    data            = read_csv(filename_climatology, path_file)
    
    print('files csv read')

    # assign color label to each data based on its month
    colors, month_to_color = add_color_label_month(data)
    
    # converting data to xarray
    data = data.to_xarray()
    
    # add colors variable to data xarray dataset
    data = data.assign(colors=(['index'], colors))   

    # select data in the domain and only hail and precipitation
    domain_plot = det_domain(domain_dfg)
    data_domain = select_data_domain(domain_plot, data)
    print((data_domain))
    
    
    # call map visualization
    map_visualization(data_domain, domain_plot, month_to_color, raster_filename, path_figs, 'True')

    plot_2d_histogram(domain_plot, data_domain, path_figs, "True")
    
        
def add_color_label_month(data):
    """
    code to add color label based on dictionary to the months for each event

    Args:
        data (dataframe pandas): 
    """
    # Convert 'datetime' column to datetime type
    data.loc[:, 'TIME_EVENT'] = pd.to_datetime(data['TIME_EVENT'])
    #hail_data.loc[:, 'TIME_EVENT'] = pd.to_datetime(hail_data['TIME_EVENT'])

    # Extract month
    data['month'] = data['TIME_EVENT'].dt.month
    
    
    # Generate a list of colors
    month_to_color = {month: colors_list[month % len(colors_list)] for month in range(1, 13)}
    
    # assign color string via mapping
    colors = data['month'].map(month_to_color).tolist()
        
    return colors, month_to_color


def plot_2d_histogram(domain, data, path_figs, plot_city):
    """
    function to plot 2d density histogram for hail and precip events
    

    Args:
        - data (xarray dataset): dataset containing hail/rain events
        - domain (tuple): A tuple containing the latitude and longitude boundaries of the main domain in the form (minlat, maxlat, minlon, maxlon).
        - path_figs: string containing output path
        - plot_city: boolean variable, when true plots location of cities
    """

    # set up plot
    fig = plt.figure(figsize=(10, 8))
    
    # Extract main domain boundaries
    minlat_main, maxlat_main, minlon_main, maxlon_main = domain
    
    ax = plt.axes(projection=ccrs.PlateCarree())
        
    # defining lat lon grid
    lats = np.linspace(minlat_main, maxlat_main, 5)
    lons = np.linspace(minlon_main, maxlon_main, 7)
    distr_2d = np.zeros((len(lats), len(lons)))
    
    print(lats, lons)
    # counting number of events per bin of lats/lons
    for i_lats, lat in enumerate(lats[:-1]):
        for i_lons, lon in enumerate(lons[:-1]):

            i_bin = np.where((data.LATITUDE.values > lat) * \
                            (data.LATITUDE.values > lats[i_lats+1]) * \
                            (data.LONGITUDE.values > lon) * \
                            (data.LONGITUDE.values < lons[i_lons+1]))[0]
            if len(i_bin) > 0:
                distr_2d[i_lats, i_lons] = len(i_bin)
                print(lat, lats[i_lats+1])
                print(lon, lons[i_lons+1])
                print(len(i_bin))
                
    #distr_norm = normalize_2dhist(distr_2d)

    plt.pcolormesh(lons, 
                lats, 
                distr_2d, 
                transform=ccrs.PlateCarree(),
                cmap=CMAP, 
                )
    ax.coastlines()

    
    plt.savefig(
        os.path.join(path_figs, "2d_histogram_ESSL_events_density.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        )

    
def select_data_domain(domain, data):
    """
    function to :
        - convert time stamps into datetime and add a variable for months
        - select lat lons of the hail and prec data in the domain
        - selects with respect to summer months
    Args:
        domain_plot (list): list of lats lons defining the domain
        data (xarray dataset): xarray dataset containing the ESSL events, lats, lons, quality flags
    
    returns
        dataset (xarray dataset): selection of hail and prec events within the domain
    """
    
    # Convert 'datetime' column to datetime type
    datetime = pd.to_datetime(data.TIME_EVENT.values)
    
    # select month string from datetime timestamps
    months = datetime.month

    # add month variable to the dataset
    data = data.assign(months=(['index'], months))

    print(len(data.index.values))
    
    # selecting hail and precip data
    ind_sel_events = np.where(np.logical_or(data.TYPE_EVENT.values == 'HAIL', data.TYPE_EVENT.values == 'PRECIP'))[0]
    data_events = data.sel(index=ind_sel_events)
    print(len(data_events.index.values))

    # reading max min lats lons from the domain list
    minlat, maxlat, minlon, maxlon = domain
    
    # selecting based on domain size from the subset of events
    ind_sel = np.where((data_events.LATITUDE.values > minlat) *\
        (data_events.LATITUDE.values <= maxlat) * \
            (data_events.LONGITUDE.values > minlon) * \
                (data_events.LONGITUDE.values <= maxlon))[0]

    data_selection = data_events.isel(index=ind_sel)
    print(len(data_selection.index.values))
    
    # selecting only months between april and october
    ind_months = np.where((data_selection.months.values > 3) * (data_selection.months.values < 10))[0]
    data_sel_summer = data_selection.isel(index=ind_months)
    
    return data_sel_summer
    

def normalize_2dhist(data):
    """
    Normalize CFAD given in absolute counts by max value occurring in the 
    bins
 
    Parameters
    ----------
    data: 2d occurrences in absolute counts

    Returns
    -------
    data_norm: normalized data over the max value over the dims
    """

    data_norm = data / np.nanmax(data)

    return data_norm

def map_visualization(data, domain, month_to_color, raster_filename, path_figs, plot_city='True'):
    """
    Plot geographic event data on a map.

    This function visualizes the geographical distribution of events within specified domains on a map. 
    The map is overlaid with raster data for contextual geographical information. Events are categorized by type 
    and colored by the month of occurrence. Optionally, major city locations can be plotted on the map. 
    The function saves the generated plot as a PNG file.

    Parameters:
    - data (xarray dataset): dataset containing hail/rain events
    - domain (tuple): A tuple containing the latitude and longitude boundaries of the main domain in the form (minlat, maxlat, minlon, maxlon).
    - raster_filename (str): Filename of the raster file to be used as a background map.
    - title (str): The title for the map plot.
    - path_file (str): Path to the directory containing the raster file.
    - path_figs (str): Path to the directory where the plot image will be saved.
    - plot_city (bool): A flag to indicate whether to plot major city locations (True) or not (False).
    """
    # define title and domain
    time_period = '2000-2024'
    title = 'ESSL Events Location'
    

    # Extract main domain boundaries
    minlat_main, maxlat_main, minlon_main, maxlon_main = domain

    with rasterio.open(raster_filename) as src:
        
        # Create a map plot
        fig, ax = plt.subplots(figsize=(12, 8),subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([minlon_main, maxlon_main, minlat_main, maxlat_main])
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        ax.imshow(src.read(1), origin='upper', cmap='binary', extent=extent, transform=ccrs.PlateCarree(), alpha=0.5, interpolation='spline36')
        
        ax.add_feature(cfeature.OCEAN, color='blue')

        # Add lat-lon axis tick labels
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        if plot_city:
            #add majot city coordinates
            trento = [46.0667, 11.1167] #lat, lon
            bolzano = [46.4981, 11.3548]
            Penegal = [46.43921, 11.2155]
            Tarmeno = [46.34054, 11.2545]
            Vilpiano = [46.55285, 11.20195]
            Sarntal = [46.56611, 11.51642]
            Cles_Malgolo = [46.38098, 11.08136]
            
            # Plot the points
            ax.scatter(trento[1], trento[0], marker='x', color='black', s=50, transform=ccrs.PlateCarree())
            ax.scatter(bolzano[1], bolzano[0], marker='x', color='black', s=50, transform=ccrs.PlateCarree())
            ax.scatter(Penegal[1], Penegal[0], marker='x', color='black', s=50, transform=ccrs.PlateCarree())
            ax.scatter(Cles_Malgolo[1], Cles_Malgolo[0], marker='x', color='black', s=50, transform=ccrs.PlateCarree())                        
            ax.scatter(Tarmeno[1], Tarmeno[0], marker='x', color='black', s=50, transform=ccrs.PlateCarree())            
            ax.scatter(Vilpiano[1], Vilpiano[0], marker='x', color='black', s=50, transform=ccrs.PlateCarree())            
            ax.scatter(Sarntal[1], Sarntal[0], marker='x', color='black', s=50, transform=ccrs.PlateCarree())            

            # Plot the names next to the points, adjusted for lower right positioning
            ax.text(trento[1] + 0.02, trento[0] - 0.02, 'Trento', color='black', transform=ccrs.PlateCarree(), ha='left', va='top')
            ax.text(bolzano[1] + 0.02, bolzano[0] - 0.02, 'Bolzano', color='black', transform=ccrs.PlateCarree(), ha='left', va='top')
            ax.text(Penegal[1] + 0.02, Penegal[0] - 0.02, 'Penegal', color='black', transform=ccrs.PlateCarree(), ha='left', va='top')
            ax.text(Cles_Malgolo[1] + 0.02, Cles_Malgolo[0] - 0.02, 'Cles_Malgolo', color='black', transform=ccrs.PlateCarree(), ha='left', va='top')
            ax.text(Tarmeno[1] + 0.02, Tarmeno[0] - 0.02, 'Tarmeno', color='black', transform=ccrs.PlateCarree(), ha='left', va='top')
            ax.text(Vilpiano[1] + 0.02, Vilpiano[0] - 0.02, 'Vilpiano', color='black', transform=ccrs.PlateCarree(), ha='left', va='top')
            ax.text(Sarntal[1] + 0.02, Sarntal[0] - 0.02, 'Sarntal', color='black', transform=ccrs.PlateCarree(), ha='left', va='top')

        # separating hail from precipitation for plotting 
        hail_data = data.isel(index=np.where(data.TYPE_EVENT.values == 'HAIL')[0])
        prec_data = data.isel(index=np.where(data.TYPE_EVENT.values == 'PRECIP')[0])
        
        # plot dots for rain
        ax.scatter(prec_data['LONGITUDE'], 
                   prec_data['LATITUDE'],
                   marker='o', 
                   label='PREC', 
                   c=prec_data.colors.values, 
                   alpha=0.7, 
                   transform=ccrs.PlateCarree())
        
        # plot dots for hail
        ax.scatter(hail_data['LONGITUDE'], 
                   hail_data['LATITUDE'], 
                   marker='^', 
                   label='HAIL', 
                   c=hail_data['colors'].values, 
                   alpha=0.7,
                   transform=ccrs.PlateCarree())
        
        ax.legend(frameon="False", loc='center left', bbox_to_anchor=(1, 0.7))
        
        # Get unique months present in the dataframes
        unique_months = np.unique(data['months'].values)
        
        # Custom legend handles for the unique months
        month_labels = [
            "January", "February", "March", "April", "May", "June", 
            "July", "August", "September", "October", "November", "December"
        ]

        handles = [mpatches.Patch(color=month_to_color[month], label=month_labels[month-1]) 
                        for month in unique_months]

        # Combine handles and labels
        #handles = [rain_legend, hail_legend] + month_handles
        labels = [h.get_label() for h in handles]

        # add title
        ax.set_title(title+' - '+time_period,fontsize=14,fontweight='bold')

        # Display the combined legend outside the plot on the central right
        ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
        
        
        fig.savefig(path_figs+title+'_'+time_period+'.png',bbox_inches='tight')

    
    
    
    



if __name__ == "__main__":
    main()