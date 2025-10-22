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
import matplotlib.gridspec as gridspec
import os
import imageio
import seaborn as sns
from scipy.stats import gaussian_kde


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



def plot_intensity_distributions(df, event_type, quantity_name, unit, save_path=None):
    """
    Plot distributions of precipitation and hail intensities.
    """

    # Filter valid values
    df_event = df[(df['TYPE_EVENT'] == event_type)]   

    df_event['TIME_EVENT'] = pd.to_datetime(df_event['TIME_EVENT'], errors='coerce', utc=True)

    df_event['month'] = df_event['TIME_EVENT'].dt.month

    df_event = df_event[(df_event['month'] >= 4) & (df_event['month'] <= 9)] 

    #remove invalid valuues (negative values and nans)
    df_event = df_event[df_event[quantity_name] >= 0]
    df_event = df_event[df_event[quantity_name].notna()]

    # Create plot
    plt.figure(figsize=(7, 4))
    sns.set_theme(style="whitegrid")

    if not df_event.empty:
        sns.histplot(df_event[quantity_name], bins=25, kde=False, color='dodgerblue', alpha=0.6)#, label=f'{quantity_name} ({unit})')

    plt.xlabel(f"{quantity_name} ({unit})")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {quantity_name} ({unit})")
    #log
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path+f"distribution_{quantity_name}.png", dpi=300, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")





def plot_events(datasets, main_domain, subdomains, type, raster_filename, title, path_file, path_figs, plot_city, intensity_threshold):
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

    with rasterio.open(raster_filename) as src:
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

            #get the name of the intensity
            if type=='PRECIP':
                name_variable = 'PRECIPITATION_AMOUNT'
            elif type == 'HAIL':
                name_variable = 'MAX_HAIL_DIAMETER'
 
            # Filter events based on type and intensity
            data = filtered_data[(filtered_data['TYPE_EVENT'] == type) & 
                                                (filtered_data[name_variable] > intensity_threshold)]
            #data = filtered_data[filtered_data['TYPE_EVENT'] == type ] #'PRECIP']

            # get hail events
            #hail_data = filtered_data[filtered_data['TYPE_EVENT'] == 'HAIL']

            # Convert 'TIME_EVENT' column to datetime type with a specific format
            data['TIME_EVENT'] = pd.to_datetime(data['TIME_EVENT'], errors='coerce', utc=True)

            #data.loc[:, 'TIME_EVENT'] = pd.to_datetime(data['TIME_EVENT'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

            # After ensuring conversion, extract the month
            if pd.api.types.is_datetime64_any_dtype(data['TIME_EVENT']):
                data['month'] = data['TIME_EVENT'].dt.month
            else:
                # Handle the case where 'TIME_EVENT' did not convert to datetime successfully
                print("Conversion to datetime failed or 'TIME_EVENT' does not contain datetime values.")

            #only keep months (April to September)
            data = data[(data['month'] >= 4) & (data['month'] <= 9)]
            
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

            #colors = data['month'].map(month_to_color).tolist()
            #hail_colors = hail_data['month'].map(month_to_color).tolist()

            # Verify 'month' column creation
            if 'month' in data.columns:
                # Proceed with mapping and other operations
                colors = data['month'].map(month_to_color).tolist()
            else:
                print("The 'month' column does not exist in the DataFrame.")
            
            ax.scatter(data['LONGITUDE'], data['LATITUDE'], 
            marker='o', label='RAIN', c=colors, alpha=0.7,
            transform=ccrs.PlateCarree())

            #ax.scatter(hail_data['LONGITUDE'], hail_data['LATITUDE'], 
            #marker='^', label='HAIL', c=hail_colors, alpha=0.7,
            #transform=ccrs.PlateCarree())
        
        # Get unique months present in the dataframes
        unique_months = np.sort(data['month'].unique())
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




def plot_events_with_density(datasets, main_domain, subdomains, type, raster_filename,
                             title, path_file, path_figs, plot_city, intensity_threshold):
    """
    Plot event density (filled contour) + overlay high-intensity events as dots.
    """

    minlat_main, maxlat_main, minlon_main, maxlon_main = main_domain

    with rasterio.open(raster_filename) as src:
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([minlon_main, maxlon_main, minlat_main, maxlat_main])
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)
        ax.imshow(src.read(1), origin='upper', cmap='gist_earth',
                  extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top],
                  transform=ccrs.PlateCarree(), alpha=0.5, interpolation='spline36')
        ax.add_feature(cfeature.OCEAN, color='blue')
        # Collect all events to compute density later
        all_lats, all_lons = [], []

        for i, domain in enumerate(subdomains):
            minlat, maxlat, minlon, maxlon = domain
            data = datasets[i]
            data = data[(data['LATITUDE'] >= minlat) & (data['LATITUDE'] <= maxlat) &
                        (data['LONGITUDE'] >= minlon) & (data['LONGITUDE'] <= maxlon)]

            # Intensity field
            if type == 'PRECIP':
                intensity_field = 'PRECIPITATION_AMOUNT'
                unit = 'mm'
            elif type == 'HAIL':
                intensity_field = 'MAX_HAIL_DIAMETER'
                unit = 'cm'
            else:
                raise ValueError("type must be 'PRECIP' or 'HAIL'")

            data = data[data['TYPE_EVENT'] == type]
            data = data.dropna(subset=['LATITUDE', 'LONGITUDE'])

            data['TIME_EVENT'] = pd.to_datetime(data['TIME_EVENT'], errors='coerce', utc=True)

            data['month'] = data['TIME_EVENT'].dt.month

            data = data[(data['month'] >= 4) & (data['month'] <= 9)]

            # Append all for density
            all_lats.extend(data['LATITUDE'])
            all_lons.extend(data['LONGITUDE'])

            # Filter strong events only
            strong_events = data[data[intensity_field] > intensity_threshold]

            # Overlay thresholded dots
            # ax.scatter(strong_events['LONGITUDE'], strong_events['LATITUDE'],
            #            s=30, c='blue', marker='o', edgecolor='black',
            #            alpha=0.9, transform=ccrs.PlateCarree(), label=f">{intensity_threshold} {unit}")

        # ---- Compute and plot density map ----
        if len(all_lons) > 50:  # at least 50 points to get smooth KDE
            xi, yi = np.mgrid[minlon_main:maxlon_main:200j, minlat_main:maxlat_main:200j]
            coords = np.vstack([all_lons, all_lats])
            kde = gaussian_kde(coords, bw_method=0.05)
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
            zi = zi.reshape(xi.shape)

            #cf = ax.contourf(xi, yi, zi, levels=15, cmap='Reds',alpha=0.8, transform=ccrs.PlateCarree())
            # Mask zero or near-zero values to make them transparent
            zi_masked = np.ma.masked_where(zi <= 0.01, zi)

            cf = ax.contourf(
                xi, yi, zi_masked,
                levels=15,
                cmap='Reds',
                alpha=0.8,
                transform=ccrs.PlateCarree()
            )


            cbar = plt.colorbar(cf, ax=ax, orientation='vertical', shrink=0.6, pad=0.02)
            cbar.set_label("Event Density")

        # ---- Optional: Add city markers ----
        if plot_city:
            cities = {"Trento": [46.0667, 11.1167], "Bolzano": [46.4981, 11.3548]}
            for name, (lat, lon) in cities.items():
                ax.scatter(lon, lat, marker='x', color='black', s=50, transform=ccrs.PlateCarree())
                ax.text(lon + 0.05, lat - 0.05, name, color='black', transform=ccrs.PlateCarree())

        ax.set_title(f"{title} - {type}", fontsize=13, fontweight='bold')
                     #\nDensity + Strong Events > {intensity_threshold} {unit}",
                     
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(path_figs + title + f"_{type}_density.png", bbox_inches='tight', dpi=300)
        



def plot_monthly_event_frequency(data, domain, title, path_figs):
    """
    Plots a bar chart representing the monthly frequency of precipitation and hail events within a specified geographical domain.

    Parameters:
    - data (pd.DataFrame): The complete dataset containing the event data.
    - domain (tuple): A tuple containing the latitude and longitude boundaries (minlat, maxlat, minlon, maxlon) to filter the events by geographical location.
    - title (str): The title for the plot. This title will also be used in the filename when saving the figure.
    - path_figs (str): The path to the directory where the figure will be saved. Should end with a forward slash (/).

    The function filters the dataset for events within the specified geographical domain, separates the precipitation and hail events, and groups them by month to calculate the frequency. 
    It then plots these frequencies as side-by-side bars for each month of the year. The x-axis represents the months, and the y-axis shows the number of events. 
    The plot is saved as a PNG file using the provided title in the specified directory.

    Returns:
    - None
    """
    minlat, maxlat, minlon, maxlon = domain

    data['TIME_EVENT'] = pd.to_datetime(data['TIME_EVENT'], errors='coerce', utc=True)

    data['month'] = data['TIME_EVENT'].dt.month

    data = data[(data['month'] >= 4) & (data['month'] <= 9)]

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

    #print tot counts for rain and hail
    total_rain = monthly_rain['PRECIP'].sum()
    total_hail = monthly_hail['HAIL'].sum()
    print(f'Total PRECIP events: {total_rain}')
    print(f'Total HAIL events: {total_hail}')

    # Merge the monthly data and fill missing months with 0
    all_months = pd.DataFrame({'month': range(4, 10)})
    monthly_data = all_months.merge(monthly_rain, on='month', how='left').merge(monthly_hail, on='month', how='left').fillna(0)

    # Define positions for the bars
    bar_width = 0.35
    rain_positions = np.arange(len(monthly_data['month'])) - bar_width / 2
    hail_positions = np.arange(len(monthly_data['month'])) + bar_width / 2

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot the bars side by side
    ax.bar(rain_positions, monthly_data['PRECIP'], width=bar_width, label='PRECIP', color='#1f77b4')
    ax.bar(hail_positions, monthly_data['HAIL'], width=bar_width, label='HAIL', color='#ff7f0e')
    
    # Set the x-axis labels to be the month names and rotate them for better readability
    # month_ticks = [
    #     "January", "February", "March", "April", "May", "June", 
    #     "July", "August", "September", "October", "November", "December"
    # ]
    month_ticks = [
        "April", "May", "June", 
        "July", "August", "September"
    ]
    ax.set_xticks(range(0, 6))
    ax.set_xticklabels(month_ticks, rotation=45, fontsize=16)
    # Set larger font size for y-tick labels
    ax.tick_params(axis='y', labelsize=16)
    
    # Customize the labels and title with larger font sizes
    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel('Number of Events', fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=16)
    
    # Save the plot
    fig.savefig(path_figs + title + '.png', bbox_inches='tight')



def plot_event_trend(data, domain, title, path_figs, time_period, frequency='W'):
    """
    Plots a trend of event occurrences over time for specified event types within a given geographical domain.

    Parameters:
    - data (pd.DataFrame): A DataFrame containing the event data with columns for 'LATITUDE', 'LONGITUDE', 'TIME_EVENT', and 'TYPE_EVENT'.
    - domain (tuple): A tuple of four floats representing the minimum latitude, maximum latitude, minimum longitude, and maximum longitude that define the geographical domain of interest.
    - title (str): The title of the plot, which will also be used in the filename when saving the figure.
    - path_figs (str): The directory path where the figure will be saved. It should end with a slash ('/').
    - time_period (tuple): A tuple of two strings representing the start and end dates for the time range of interest, in the format 'YYYY-MM-DD'.
    - frequency (str, optional): A string that defines the frequency of the time period aggregation, defaulting to 'W' for weekly. Can also be 'D' for daily.

    The function filters the data based on the specified geographical domain, resamples the data based on the provided frequency, counts the number of events within each time period, and plots the trend for 'PRECIP' and 'HAIL' events. 
    The x-axis represents the time (either as date or week of the year), and the y-axis represents the number of events. 
    It uses different colors and markers to distinguish between the event types. The resulting plot is displayed and also saved as a PNG file to the specified path.

    Returns:
    - None
    """
    minlat, maxlat, minlon, maxlon = domain
    filtered_data = data[(data['LATITUDE'] >= minlat) & (data['LATITUDE'] <= maxlat) &
                         (data['LONGITUDE'] >= minlon) & (data['LONGITUDE'] <= maxlon)]
    
    # Convert 'datetime' column to datetime type
    filtered_data['TIME_EVENT'] = pd.to_datetime(filtered_data['TIME_EVENT'])

    date_format = '%Y-%m-%d'
    
    # Resample the data according to the specified frequency
    if frequency.upper() == 'W':
        # Extract week number and year for weekly frequency
        filtered_data['time_period'] = filtered_data['TIME_EVENT'].dt.strftime('%Y-%U')
        time_format = '%Y-%U'
        freq = 'W-MON'
    elif frequency.upper() == 'D':
        # Extract date for daily frequency
        filtered_data['time_period'] = filtered_data['TIME_EVENT'].dt.date
        time_format = '%Y-%m-%d'
        freq = 'D'
    
    # Create a full range of time periods for 2021 and 2022
    all_time_periods = pd.date_range(start=time_period[0], end=time_period[1], freq=freq).strftime(time_format).unique()
    all_time_periods_df = pd.DataFrame(all_time_periods, columns=['time_period'])

    # Group by time period and count events
    event_data = filtered_data.groupby(['time_period', 'TYPE_EVENT']).size().unstack(fill_value=0).reset_index()

    # Merge with the all_time_periods dataframe to include periods with zero events
    event_data = all_time_periods_df.merge(event_data, on='time_period', how='left').fillna(0)

    # Ensure the order is correct after merging
    event_data['time_period'] = pd.to_datetime(event_data['time_period'] + ('-1' if frequency.upper() == 'W' else ''), format=time_format + ('-%w' if frequency.upper() == 'W' else ''))
    event_data.sort_values('time_period', inplace=True)
    event_data['time_label'] = event_data['time_period'].dt.strftime(date_format)

    # Plotting
    fig, ax = plt.subplots(figsize=(17, 8))
    
    # Plot the lines
    ax.plot(event_data['time_label'], event_data['PRECIP'], label='PRECIP', color='#1f77b4', marker='o')
    ax.plot(event_data['time_label'], event_data['HAIL'], label='HAIL', color='#ff7f0e', marker='x')
    
    # Calculate the correct tick positions
    tick_positions = np.arange(len(event_data['time_label']))

    # Apply the calculated tick positions
    ax.set_xticks(tick_positions)

    # Set the labels with the adjusted tick positions
    ax.set_xticklabels(event_data['time_label'], rotation=60, fontsize=10, ha='right')

    # Ensure that the tick marks and labels are aligned correctly
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='y', labelsize=12) 

    # Customize the labels and title with larger font sizes
    ax.set_xlabel('Date' if frequency.upper() == 'D' else 'Week of the Year', fontsize=14)
    ax.set_ylabel('Number of Events', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=14)
    
    # Save the plot
    plt.tight_layout()
    plt.show()
    fig.savefig(path_figs + title + '.png', bbox_inches='tight')




def plot_intensity_trend(data, domain, title, path_figs, time_period, frequency='W'):
    """
    Plots a trend of PRECIP intensity and HAIL intensity over time with dual y-axes.
    
    Parameters:
    - data (pd.DataFrame): The complete dataset containing the event data.
    - domain (tuple): A tuple containing the latitude and longitude boundaries to filter events.
    - title (str): The title for the plot.
    - path_figs (str): The directory path for saving the plot image.
    - time_period (tuple): The start and end of the time period to plot (YYYY-MM-DD format).
    - frequency (str): The frequency of the data aggregation ('W' for weekly, 'D' for daily).
    
    The function filters the dataset for events within the specified geographical domain, resamples the data to the specified frequency, and calculates the mean intensity for precipitation and hail. 
    It then creates a plot with two y-axes, one for precipitation amount in mm and another for maximum hail diameter in cm, across the specified time period.
    
    Returns:
    - None
    """
    minlat, maxlat, minlon, maxlon = domain
    filtered_data = data[(data['LATITUDE'] >= minlat) & (data['LATITUDE'] <= maxlat) &
                         (data['LONGITUDE'] >= minlon) & (data['LONGITUDE'] <= maxlon)]
    
    # Convert 'datetime' column to datetime type
    filtered_data['TIME_EVENT'] = pd.to_datetime(filtered_data['TIME_EVENT'])

    # Define date_format outside of the conditionals
    date_format = '%Y-%m-%d'  # This will be used for both weekly and daily labels
    
    # Resample the data according to the specified frequency
    if frequency.upper() == 'W':
        filtered_data['time_period'] = filtered_data['TIME_EVENT'].dt.strftime('%Y-%U')
        time_format = '%Y-%U'
        freq = 'W-MON'
    elif frequency.upper() == 'D':
        filtered_data['time_period'] = filtered_data['TIME_EVENT'].dt.date
        time_format = '%Y-%m-%d'
        freq = 'D'
    
    # Group by time period and sum intensities for each event type
    intensity_data = filtered_data.groupby(['time_period']).agg({
        'PRECIPITATION_AMOUNT': 'mean',  # Take the mean of precipitation amounts
        'MAX_HAIL_DIAMETER': 'mean'      # Take the mean of hail diameters
    }).reset_index()

    # Create a full range of time periods for the specified time_period
    all_time_periods = pd.date_range(start=time_period[0], end=time_period[1], freq=freq).strftime(time_format).unique()
    all_time_periods_df = pd.DataFrame(all_time_periods, columns=['time_period'])

    # Merge with the all_time_periods dataframe to include periods with zero intensity
    intensity_data = all_time_periods_df.merge(intensity_data, on='time_period', how='left').fillna(0)

    # Ensure the order is correct after merging
    intensity_data['time_period'] = pd.to_datetime(intensity_data['time_period'] + ('-1' if frequency.upper() == 'W' else ''), format=time_format + ('-%w' if frequency.upper() == 'W' else ''))
    intensity_data.sort_values('time_period', inplace=True)
    intensity_data['time_label'] = intensity_data['time_period'].dt.strftime(date_format)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(17, 8))
    
    # Plot the PRECIP intensity on the left y-axis
    ax1.plot(intensity_data['time_label'], intensity_data['PRECIPITATION_AMOUNT'], color='#1f77b4', marker='o')
    ax1.set_xlabel('Date' if frequency.upper() == 'D' else 'Week of the Year', fontsize=14)
    ax1.set_ylabel('PRECIPITATION AMOUNT (mm)', fontsize=14, color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4', color='#1f77b4', labelsize = 14)

    # Create a twin axis for the HAIL intensity
    ax2 = ax1.twinx()  
    ax2.plot(intensity_data['time_label'], intensity_data['MAX_HAIL_DIAMETER'], color='#ff7f0e', marker='x')
    ax2.set_ylabel('MAX HAIL DIAMETER (cm)', fontsize=14, color= '#ff7f0e' )
    ax2.tick_params(axis='y', labelcolor='#ff7f0e', color='#ff7f0e', labelsize = 14)

    # Set the x-axis tick positions and labels
    tick_positions = np.arange(len(intensity_data['time_label']))
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(intensity_data['time_label'], rotation=60, fontsize=10, ha='right')
    ax1.tick_params(axis='x', which='major', labelsize=10)

    # Title and legend
    ax1.set_title(title, fontsize=16, fontweight='bold')
    
    # Save and show plot
    plt.tight_layout()
    plt.show()
    fig.savefig(path_figs + title + '.png', bbox_inches='tight')



def plot_top_intensities(data, domain, type, n_events, title, path_figs, path_raster):
    """
    Generates and saves a set of plots visualizing the top weather event intensities within a specified domain.

    The function filters the data for the given domain, selects the top 20 events of precipitation and hail based on intensity,
    and generates a set of bar plots for these events. Additionally, it displays the top events on a geographical map.
    Parameters:
    - data (DataFrame): The dataset containing weather event information including type, intensity, and coordinates.
    - domain (tuple): A tuple specifying the latitude and longitude bounds (minlat, maxlat, minlon, maxlon) for filtering the data.
    - type (str): 'PRECIP' or 'HAIL'
    - n_events (int): Number of events to show in the bar plot
    - title (str): The title for the plots, which also serves as the filename when saving.
    - path_figs (str): The file path where the generated plot image will be saved.
    - path_raster (str): The file path to a raster file that provides the base map for plotting events geographically.

    Returns:
    - None: This function does not return any value.
    """

    if type == 'PRECIP':
        quantity = 'PRECIPITATION_AMOUNT'
        label_name = 'Precipitation Amount (mm)'
        marker_size = 2
    elif type == 'HAIL':
        quantity = 'MAX_HAIL_DIAMETER'
        label_name = 'Max Hail Diameter (cm)'
        marker_size = 20

    minlat, maxlat, minlon, maxlon = domain
    filtered_data = data[(data['LATITUDE'] >= minlat) & (data['LATITUDE'] <= maxlat) &
                         (data['LONGITUDE'] >= minlon) & (data['LONGITUDE'] <= maxlon)]

    # Convert 'datetime' column to datetime type
    filtered_data['TIME_EVENT'] = pd.to_datetime(filtered_data['TIME_EVENT'])

    # Filter top events based on intensity
    top_events = filtered_data[filtered_data['TYPE_EVENT'] == type].nlargest(n_events, quantity).reset_index()

    # Assign a unique color to each day
    unique_days = np.unique(top_events['TIME_EVENT'].dt.date)
    color_map = plt.cm.get_cmap('tab10', len(unique_days))  # Use a colormap with as many colors as unique days
    day_to_color = {day: color_map(i) for i, day in enumerate(unique_days)}  # Map each day to a color

    # Create figure with a specified size
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

    # Bar plot
    ax1 = plt.subplot(gs[0, 0])
    for i, row in top_events.iterrows():
        color = day_to_color[row['TIME_EVENT'].date()]
        ax1.barh(i, row[quantity], color=color, alpha=0.6)
    ax1.set_xlabel(label_name, fontsize=14)
    ax1.set_yticks(top_events.index)
    ax1.set_yticklabels(top_events['TIME_EVENT'].dt.strftime('%Y-%m-%d %H:%M'), fontsize=12)
    #ax1.set_title(f'Top {n_events} {type} Events', fontsize=14)
    ax1.tick_params(axis='x', which='major', labelsize=12)
    ax1.set_ylim(len(top_events) - 0.5, -0.5)  # Invert the y-axis

    # Map plot
    ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
    # Plot raster
    with rasterio.open(path_raster) as src:
        ax2.set_extent([minlon, maxlon, minlat, maxlat])
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        ax2.add_feature(cfeature.COASTLINE)
        ax2.add_feature(cfeature.LAKES, alpha=0.5)
        ax2.add_feature(cfeature.RIVERS)
        ax2.add_feature(cfeature.OCEAN, color='blue')

        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        ax2.imshow(src.read(1), origin='upper', cmap='gist_earth', extent=extent, transform=ccrs.PlateCarree(), alpha=0.5, interpolation='spline36')

    # Plot top events on the map with colored markers
    for i, row in top_events.iterrows():
        color = day_to_color[row['TIME_EVENT'].date()]
        size = row[quantity]  # Adjust this scaling as necessary for your dataset
        ax2.scatter(row['LONGITUDE'], row['LATITUDE'], s=size*marker_size, color=color, edgecolor='black', zorder=3, transform=ccrs.Geodetic())

    # Adding a custom legend for the dates
    patches = [mpatches.Patch(color=color, label=str(day)) for day, color in day_to_color.items()]
    ax2.legend(handles=patches, loc='upper left', fontsize=12, title="Event Dates")

    # Adjust the title space and layout of the plots
    fig.suptitle(title+' - '+label_name, fontsize=16, fontweight='bold')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, wspace=0.1, hspace=0.1)

    # Save the figure
    plt.savefig(f"{path_figs}/{title.replace(' ', '_').lower()}-{type}.png", bbox_inches='tight')

    # Show the figure
    plt.show()




def save_daily_event_maps(data, domain, title, path_raster, output_dir, year, start_month=5, end_month=9):
    """
    Saves daily weather event maps to an output directory.
    """
    minlat, maxlat, minlon, maxlon = domain
    data = data[(data['LATITUDE'] >= minlat) & (data['LATITUDE'] <= maxlat) &
                         (data['LONGITUDE'] >= minlon) & (data['LONGITUDE'] <= maxlon)]

    # Filter data for the specified year and date range
    data['TIME_EVENT'] = pd.to_datetime(data['TIME_EVENT'])
    data = data[(data['TIME_EVENT'].dt.year == year) &
                (data['TIME_EVENT'].dt.month >= start_month) &
                (data['TIME_EVENT'].dt.month <= end_month)]

    # Get maximum intensity for normalization
    max_precip_intensity = data[data['TYPE_EVENT'] == 'PRECIP']['PRECIPITATION_AMOUNT'].max()
    max_hail_intensity = data[data['TYPE_EVENT'] == 'HAIL']['MAX_HAIL_DIAMETER'].max()

    # Define the normalization for the colorbars
    norm_precip = mcolors.Normalize(vmin=0, vmax=max_precip_intensity)
    norm_hail = mcolors.Normalize(vmin=0, vmax=max_hail_intensity)

    # Create a date range for the specified months
    date_range = pd.date_range(start=f'{year}-{start_month}-01', end=f'{year}-{end_month}-30')

    for current_date in date_range:
        daily_data = data[data['TIME_EVENT'].dt.date == current_date.date()]

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Plot the base map with the raster background
        with rasterio.open(path_raster) as src:
            # Create a map plot
            ax.set_extent([minlon, maxlon, minlat, maxlat])
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAKES, alpha=0.5, color='dimgray')
            ax.add_feature(cfeature.RIVERS, color='dimgray')

            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            ax.imshow(src.read(1), origin='upper', cmap='gist_gray', extent=extent, transform=ccrs.PlateCarree(), alpha=0.5, interpolation='spline36')
            
            ax.add_feature(cfeature.OCEAN, color='dimgray')

            # Add lat-lon axis tick labels
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 12, 'color': 'black'}
            gl.ylabel_style = {'size': 12, 'color': 'black'}

        # Plot each event
        for _, row in daily_data.iterrows():
            if row['TYPE_EVENT'] == 'PRECIP':
                marker = 'o'  # Example marker for precipitation
                cmap = plt.get_cmap('winter_r') #plt.cm.Blues  # Color map for precipitation
                norm = norm_precip  # Normalization based on precipitation
                color = cmap(norm(row['PRECIPITATION_AMOUNT']))
            else:
                marker = '^'  # Example marker for hail
                cmap = plt.get_cmap('autumn_r') #plt.cm.Reds  # Color map for hail
                norm = norm_hail  # Normalization based on hail
                color = cmap(norm(row['MAX_HAIL_DIAMETER']))

            # Plot the event location
            ax.plot(row['LONGITUDE'], row['LATITUDE'], marker=marker, color=color, markersize=10, transform=ccrs.Geodetic())
    

        # Create a scalar mappable for the colorbar
        sm_precip = plt.cm.ScalarMappable(cmap= 'winter_r', norm=norm_precip)
        sm_hail = plt.cm.ScalarMappable(cmap= 'autumn_r' , norm=norm_hail)

        # Add the colorbars to the figure
        # Precipitation colorbar on the right
        cax_precip = fig.add_axes([0.8, 0.55, 0.03, 0.35])  # [left, bottom, width, height]
        cbar_precip = fig.colorbar(sm_precip, cax=cax_precip)
        cbar_precip.set_label('Precipitation Amount (mm)')

        # Hail colorbar on the right, below the precipitation colorbar
        cax_hail = fig.add_axes([0.8, 0.10, 0.03, 0.35])  # [left, bottom, width, height]
        cbar_hail = fig.colorbar(sm_hail, cax=cax_hail)
        cbar_hail.set_label('Max Hail Diameter (cm)')

        # Add the date as text on the figure
        date_str = current_date.strftime('%Y-%m-%d')
        #plt.title(f'{title} for {date_str}', fontsize=14)
        # Adjust the title space and layout of the plots
        fig.suptitle(f'{title} for {date_str}', fontsize=14, fontweight='bold')  # Reduce the pad to bring the title closer to the plots
        plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.1, wspace=0.1, hspace=0.1)

        # Save the figure
        fig.savefig(f'{output_dir}/{title}-{date_str}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory





def plot_event_counts_by_qc_level(data, domain, title, path_figs):
    """
    This function filters weather event data based on a geographical domain, then calculates the count
    of 'HAIL' and 'PRECIP' events, further breaking down the count by QC_LEVEL. It generates a stacked 
    vertical bar plot to display these counts, saves the plot to the specified path, and shows the plot.

    Parameters:
    - data (DataFrame): The dataset containing weather event information with 'LATITUDE', 'LONGITUDE',
                        'TYPE_EVENT', and 'QC_LEVEL' columns.
    - domain (tuple): A tuple specifying the latitude and longitude bounds 
                      (minlat, maxlat, minlon, maxlon) for filtering the data.
    - title (str): The title for the plot. It is also used to name the saved plot file.
    - path_figs (str): The file path where the generated plot image will be saved.

    Returns:
    - None: This function does not return any value.
    """
    minlat, maxlat, minlon, maxlon = domain
    data = data[(data['LATITUDE'] >= minlat) & (data['LATITUDE'] <= maxlat) &
                         (data['LONGITUDE'] >= minlon) & (data['LONGITUDE'] <= maxlon)]
    
    # Count the different values in 'QC_LEVEL' and the subtotal for each 'TYPE_EVENT'
    qc_counts = data.groupby(['TYPE_EVENT', 'QC_LEVEL']).size().unstack(fill_value=0)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create a stacked bar plot for each QC level
    bottom_values = pd.Series([0, 0], index=qc_counts.index)
    colors = ['blue', 'green', 'red', 'purple', 'orange']  # Change/add colors as needed based on the number of QC levels
    for i, qc_level in enumerate(qc_counts.columns):
        count = qc_counts[qc_level].sum()
        ax.bar(qc_counts.index, qc_counts[qc_level], bottom=bottom_values, color=colors[i], label=f'QC Level {qc_level} ({str(count)})')
        bottom_values += qc_counts[qc_level]
    
    # Set the labels and title
    ax.set_ylabel('Count')
    #ax.set_yscale('log')
    #ax.set_ylim(bottom=0)
    ax.set_title(title, fontweight='bold')
    ax.set_xticklabels(qc_counts.index, rotation=0)  # Adjust rotation if needed
    
    # Place the legend outside the plot to the right
    ax.legend(title='QC Levels', loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the figure
    fig.savefig(f"{path_figs}/{title.replace(' ', '_').lower()}.png", bbox_inches='tight')
    
    # Show the plot
    plt.show()




def create_gif_from_folder(folder_path, output_path, duration=0.5):
    """
    Create a GIF from a sequence of images in a folder.

    :param folder_path: Path to the folder containing image files.
    :param output_path: Path where the GIF should be saved.
    :param duration: Duration of each frame in the GIF in seconds.
    """
    # Get file paths
    images = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.png') or file_name.endswith('.jpg'):
            file_path = os.path.join(folder_path, file_name)
            images.append(imageio.imread(file_path))
    
    # Save out as a GIF
    imageio.mimsave(output_path, images, duration=duration)



