"""
plotting function used
to analyze ESSL data
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import rasterio
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER



def plot_events_paula(datasets, main_domain, subdomains, intensity, raster_filename, title, path_file, path_figs):

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

        # # Get major cities and plot them separately
        # shapename = 'populated_places'
        # cities = shpreader.Reader(shpreader.natural_earth(resolution='50m',
        #                                                 category='cultural', name=shapename))
        # for city in cities.geometries():
        #     ax.plot(city.x, city.y, '.', color='black', markersize=5)

        # # Add physical labels (like mountain ranges)
        # physical_labels = cfeature.NaturalEarthFeature(category='cultural', 
        #                                             name='physical_labels', 
        #                                             scale='50m', 
        #                                             facecolor='none')
        # ax.add_feature(physical_labels, edgecolor='gray')

        #add majot city coordinates
        trento = [46.0667, 11.1167] #lat, lon
        bolzano = [46.4981, 11.3548]

        # Plot the points
        ax.scatter(trento[1], trento[0], color='black', s=50, transform=ccrs.PlateCarree())
        ax.scatter(bolzano[1], bolzano[0], color='black', s=50, transform=ccrs.PlateCarree())
        # Plot the names next to the points, adjusted for lower right positioning
        ax.text(trento[1] + 0.02, trento[0] - 0.02, 'Trento', color='black', transform=ccrs.PlateCarree(), ha='left', va='top')
        ax.text(bolzano[1] + 0.02, bolzano[0] - 0.02, 'Bolzano', color='black', transform=ccrs.PlateCarree(), ha='left', va='top')

        #set up labels
        num_subdomains = len(subdomains)
        label_rains = ['Rain'] + [None] * (num_subdomains - 1)
        label_hails = ['Hail'] + [None] * (num_subdomains - 1)

        for i,domain in enumerate(subdomains):
            # Filter data based on the current subdomain
            minlat, maxlat, minlon, maxlon = domain
            data = datasets[i]
            label_rain = label_rains[i]
            label_hail = label_hails[i]
            filtered_data = data[(data['lat'] >= minlat) & (data['lat'] <= maxlat) &
                                (data['lon'] >= minlon) & (data['lon'] <= maxlon)]

            # Delineate subdomains
            rect = mpatches.Rectangle((minlon, minlat), maxlon-minlon, maxlat-minlat,
                                    fill=False, edgecolor='black', linestyle='--')
            ax.add_patch(rect)

            # get rain events
            rain_data = filtered_data[filtered_data['mode'] == 'precip']

            # get hail events
            hail_data = filtered_data[filtered_data['mode'] == 'hail']

            # Convert 'datetime' column to datetime type
            rain_data.loc[:, 'datetime'] = pd.to_datetime(rain_data['datetime'])
            hail_data.loc[:, 'datetime'] = pd.to_datetime(hail_data['datetime'])
            #rain_data['datetime'] = pd.to_datetime(rain_data['datetime'])
            #hail_data['datetime'] = pd.to_datetime(hail_data['datetime'])

            # Extract month
            rain_data['month'] = rain_data['datetime'].dt.month
            hail_data['month'] = hail_data['datetime'].dt.month

            # Map each month to a color using a colormap
            colors_list = list(mcolors.TABLEAU_COLORS.values())  # use any colormap you prefer
            colors_list = ['#0048BA', '#AF002A', '#3B7A57', '#FF7E00', '#BA7890'] 
            month_to_color = {month: colors_list[month % len(colors_list)] for month in range(1, 13)}

            rain_colors = rain_data['month'].map(month_to_color).tolist()
            hail_colors = hail_data['month'].map(month_to_color).tolist()

            if intensity:
                sc1 = ax.scatter(rain_data['lon'], rain_data['lat'], 
                        c=rain_data['mm'], marker='o', label=label_rain, 
                        cmap='Blues', transform=ccrs.PlateCarree())
                
                sc2 = ax.scatter(hail_data['lon'], hail_data['lat'], 
                        c=hail_data['size (cm)'], marker='^', label=label_hail, 
                        cmap='Reds', transform=ccrs.PlateCarree())
            else:
                ax.scatter(rain_data['lon'], rain_data['lat'], 
                marker='o', label='Rain', c=rain_colors,
                transform=ccrs.PlateCarree())

                ax.scatter(hail_data['lon'], hail_data['lat'], 
                marker='^', label='Hail', c=hail_colors,
                transform=ccrs.PlateCarree())
        
        # Get unique months present in the dataframes
        unique_rain_months = rain_data['month'].unique()
        unique_hail_months = hail_data['month'].unique()

        # Combine and deduplicate
        unique_months = set(unique_rain_months).union(set(unique_hail_months))

        # Custom legend handles for rain and hail markers
        rain_legend = Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Rain')
        hail_legend = Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='Hail')

        # Custom legend handles for the unique months
        month_labels = [
            "January", "February", "March", "April", "May", "June", 
            "July", "August", "September", "October", "November", "December"
        ]

        month_handles = [mpatches.Patch(color=month_to_color[month], label=month_labels[month-1]) 
                        for month in unique_months]

        # Combine handles and labels
        handles = [rain_legend, hail_legend] + month_handles
        labels = [h.get_label() for h in handles]

        # Display the combined legend outside the plot on the central right
        ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))

        # Show colorbars at the bottom if intensity is True
        if intensity:
            cax1 = fig.add_axes([0.15, 0.08, 0.35, 0.02])
            cax2 = fig.add_axes([0.55, 0.08, 0.35, 0.02])
            
            fig.colorbar(sc1, cax=cax1, orientation='horizontal', label='Precipitation Amount (mm)')
            fig.colorbar(sc2, cax=cax2, orientation='horizontal', label='Hail Diameter (cm)')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
        fig.savefig(path_figs+title+'.png',bbox_inches='tight')



def plot_events(datasets, main_domain, subdomains, raster_filename, title, path_file, path_figs, plot_city):
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
            rain_data = filtered_data[filtered_data['TYPE_EVENT'] == 'PRECIP']

            # get hail events
            hail_data = filtered_data[filtered_data['TYPE_EVENT'] == 'HAIL']

            # Convert 'datetime' column to datetime type
            rain_data.loc[:, 'TIME_EVENT'] = pd.to_datetime(rain_data['TIME_EVENT'])
            hail_data.loc[:, 'TIME_EVENT'] = pd.to_datetime(hail_data['TIME_EVENT'])

            # Extract month
            rain_data['month'] = rain_data['TIME_EVENT'].dt.month
            hail_data['month'] = hail_data['TIME_EVENT'].dt.month

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

            rain_colors = rain_data['month'].map(month_to_color).tolist()
            hail_colors = hail_data['month'].map(month_to_color).tolist()
            
            ax.scatter(rain_data['LONGITUDE'], rain_data['LATITUDE'], 
            marker='o', label='RAIN', c=rain_colors, alpha=0.7,
            transform=ccrs.PlateCarree())

            ax.scatter(hail_data['LONGITUDE'], hail_data['LATITUDE'], 
            marker='^', label='HAIL', c=hail_colors, alpha=0.7,
            transform=ccrs.PlateCarree())
        
        # Get unique months present in the dataframes
        unique_rain_months = rain_data['month'].unique()
        unique_hail_months = hail_data['month'].unique()

        # Combine and deduplicate
        unique_months = set(unique_rain_months).union(set(unique_hail_months))

        # Custom legend handles for rain and hail markers
        rain_legend = Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='RAIN')
        hail_legend = Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='HAIL')

        # Custom legend handles for the unique months
        month_labels = [
            "January", "February", "March", "April", "May", "June", 
            "July", "August", "September", "October", "November", "December"
        ]

        month_handles = [mpatches.Patch(color=month_to_color[month], label=month_labels[month-1]) 
                        for month in unique_months]

        # Combine handles and labels
        handles = [rain_legend, hail_legend] + month_handles
        labels = [h.get_label() for h in handles]

        # add title
        ax.set_title(title,fontsize=14,fontweight='bold')

        # Display the combined legend outside the plot on the central right
        ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()
        fig.savefig(path_figs+title+'.png',bbox_inches='tight')



def plot_monthly_event_frequency(data, domain, title, path_figs):
    minlat, maxlat, minlon, maxlon = domain
    filtered_data = data[(data['LATITUDE'] >= minlat) & (data['LATITUDE'] <= maxlat) &
                         (data['LONGITUDE'] >= minlon) & (data['LONGITUDE'] <= maxlon)]
    
    # Get rain and hail events
    rain_data = filtered_data[filtered_data['TYPE_EVENT'] == 'PRECIP']
    hail_data = filtered_data[filtered_data['TYPE_EVENT'] == 'HAIL']

    # Convert 'datetime' column to datetime type and extract month
    rain_data['month'] = pd.to_datetime(rain_data['TIME_EVENT']).dt.month
    hail_data['month'] = pd.to_datetime(hail_data['TIME_EVENT']).dt.month

    # Group by month and count events
    monthly_rain = rain_data.groupby('month').size().reset_index(name='PRECIP')
    monthly_hail = hail_data.groupby('month').size().reset_index(name='HAIL')

    # Merge the monthly data and fill missing months with 0
    all_months = pd.DataFrame({'month': range(1, 13)})
    monthly_data = all_months.merge(monthly_rain, on='month', how='left').merge(monthly_hail, on='month', how='left').fillna(0)

    # Define positions for the bars
    bar_width = 0.35
    rain_positions = np.arange(len(monthly_data['month'])) - bar_width / 2
    hail_positions = np.arange(len(monthly_data['month'])) + bar_width / 2

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the bars side by side
    ax.bar(rain_positions, monthly_data['PRECIP'], width=bar_width, label='PRECIP', color='#1f77b4')
    ax.bar(hail_positions, monthly_data['HAIL'], width=bar_width, label='HAIL', color='#ff7f0e')
    
    # Set the x-axis labels to be the month names and rotate them for better readability
    month_ticks = [
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ]
    ax.set_xticks(range(0, 12))
    ax.set_xticklabels(month_ticks, rotation=45, fontsize=12)
    
    # Customize the labels and title with larger font sizes
    ax.set_xlabel('Month', fontsize=14)
    ax.set_ylabel('Number of Events', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.show()
    fig.savefig(path_figs + title + '.png', bbox_inches='tight')




