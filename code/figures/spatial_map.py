    """
    function to plot spatial map of events
    
    """

import pandas as pd

from readers import csv
from readers.filenames_paths import path_file, path_figs, filename_essl, filename_paula_grey, filename_paula_red, filename

from domain_info import teamx_domain, domain_red, domain_grey, raster_file

def main():
    
    # read csv files
    data_essl       = csv(filename_essl, path_file)
    data_paula_red  = csv(filename_paula_red, path_file)
    data_paula_grey = csv(filename_paula_grey, path_file)
    data            = csv(filename, path_file)
    
    
    # define title and domain
    time_period = '2021-2023'
    title = 'ESSL Events Location'
    
    
    # call map visualization
    map_visualization()
    
    
    
    

def map_visualization(datasets, main_domain, subdomains, type, raster_filename, title, path_file, path_figs, plot_city):
    """
    Plot geographic event data on a map.

    This function visualizes the geographical distribution of events within specified domains on a map. 
    The map is overlaid with raster data for contextual geographical information. Events are categorized by type 
    and colored by the month of occurrence. Optionally, major city locations can be plotted on the map. 
    The function saves the generated plot as a PNG file.

    Parameters:
    - datasets (list of pd.DataFrame): A list of Pandas DataFrames, each containing event data for the corresponding subdomain.
    - main_domain (tuple): A tuple containing the latitude and longitude boundaries of the main domain in the form (minlat, maxlat, minlon, maxlon).
    - subdomains (list of tuple): A list of tuples, each representing a subdomain's lat-lon boundaries as (minlat, maxlat, minlon, maxlon).
    - Type (str): 'PRECIP' or 'HAIL'
    - raster_filename (str): Filename of the raster file to be used as a background map.
    - title (str): The title for the map plot.
    - path_file (str): Path to the directory containing the raster file.
    - path_figs (str): Path to the directory where the plot image will be saved.
    - plot_city (bool): A flag to indicate whether to plot major city locations (True) or not (False).
    """

    # Extract main domain boundaries
    minlat_main, maxlat_main, minlon_main, maxlon_main = main_domain

    with rasterio.open(path_file+raster_filename) as src:
        # Create a map plot
        fig, ax = plt.subplots(figsize=(12, 8),subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([minlon_main, maxlon_main, minlat_main, maxlat_main])
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        ax.imshow(src.read(1), origin='upper', cmap='gist_earth', extent=extent, transform=ccrs.PlateCarree(), alpha=0.5, interpolation='spline36')
        
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

            # Plot the points
            ax.scatter(trento[1], trento[0], marker='x', color='black', s=50, transform=ccrs.PlateCarree())
            ax.scatter(bolzano[1], bolzano[0], marker='x', color='black', s=50, transform=ccrs.PlateCarree())
            # Plot the names next to the points, adjusted for lower right positioning
            ax.text(trento[1] + 0.02, trento[0] - 0.02, 'Trento', color='black', transform=ccrs.PlateCarree(), ha='left', va='top')
            ax.text(bolzano[1] + 0.02, bolzano[0] - 0.02, 'Bolzano', color='black', transform=ccrs.PlateCarree(), ha='left', va='top')

        for i,domain in enumerate(subdomains):
            # Filter data based on the current subdomain
            minlat, maxlat, minlon, maxlon = domain
            data = datasets[i]
            filtered_data = data[(data['LATITUDE'] >= minlat) & (data['LATITUDE'] <= maxlat) &
                                (data['LONGITUDE'] >= minlon) & (data['LONGITUDE'] <= maxlon)]

            # Delineate subdomains
            rect = mpatches.Rectangle((minlon, minlat), maxlon-minlon, maxlat-minlat,
                                    fill=False, edgecolor='black', linestyle='--')
            ax.add_patch(rect)

            # get rain events
            data = filtered_data[filtered_data['TYPE_EVENT'] == type ] #'PRECIP']

            # get hail events
            #hail_data = filtered_data[filtered_data['TYPE_EVENT'] == 'HAIL']

            # Convert 'datetime' column to datetime type
            data.loc[:, 'TIME_EVENT'] = pd.to_datetime(data['TIME_EVENT'])
            #hail_data.loc[:, 'TIME_EVENT'] = pd.to_datetime(hail_data['TIME_EVENT'])

            # Extract month
            data['month'] = data['TIME_EVENT'].dt.month
            #hail_data['month'] = hail_data['TIME_EVENT'].dt.month

            # Map each month to a color using a colormap
            #colors_list = list(mcolors.TABLEAU_COLORS.values())  # use any colormap you prefer
            colors_list = [
            '#E6194B',  # Bright Red
            '#3CB44B',  # Bright Green
            '#800000',  # Maroon 
            '#FFE119',  # Yellow
            '#911EB4',  # Purple
            '#000075',  # Dark Blue 
            '#F032E6',  # Magenta
            '#F58231',  # Orange
            '#4363D8',  # Bright Blue
            '#008080',  # Teal
            '#E6BEFF',  # Lavender
            '#46F0F0',  # Cyan
            ]

            # Generate a list of colors
            month_to_color = {month: colors_list[month % len(colors_list)] for month in range(1, 13)}

            colors = data['month'].map(month_to_color).tolist()
            #hail_colors = hail_data['month'].map(month_to_color).tolist()
            
            ax.scatter(data['LONGITUDE'], data['LATITUDE'], 
            marker='o', label='RAIN', c=colors, alpha=0.7,
            transform=ccrs.PlateCarree())

            #ax.scatter(hail_data['LONGITUDE'], hail_data['LATITUDE'], 
            #marker='^', label='HAIL', c=hail_colors, alpha=0.7,
            #transform=ccrs.PlateCarree())
        
        # Get unique months present in the dataframes
        unique_months = data['month'].unique()
        #unique_hail_months = hail_data['month'].unique()

        # Combine and deduplicate
        #unique_months = set(unique_rain_months).union(set(unique_hail_months))

        # Custom legend handles for rain and hail markers
        #rain_legend = Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='RAIN')
        #hail_legend = Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='HAIL')

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
        ax.set_title(title+' - '+type,fontsize=14,fontweight='bold')

        # Display the combined legend outside the plot on the central right
        ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
        fig.savefig(path_figs+title+'_'+type+'.png',bbox_inches='tight')

    
    
    
    
    

if __name__ == "__main__":
    main()