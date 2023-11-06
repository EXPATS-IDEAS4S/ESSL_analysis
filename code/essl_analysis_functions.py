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



def add_labels_inside(bars, data, times, latitudes, longitudes, invert_axis=False):
    for bar, amount, time, lat, lon in zip(bars, data, times, latitudes, longitudes):
        # Format the label
        label = f'{time.strftime("%Y-%m-%d")} ({lat:.2f}, {lon:.2f})'

        # Get the width and height of the bar
        bar_width = bar.get_width()
        bar_height = bar.get_height()

        # Determine the x position for text
        x_offset_fraction = 0.03  # 3% of the bar's width
        x_offset = bar_width * x_offset_fraction
        x_position = bar.get_x() + bar_width - x_offset if invert_axis else bar.get_x() + x_offset

        # The y position is the center of the bar
        y_position = bar.get_y() + bar_height / 2

        # If the axis is inverted, we need to ensure the label is still within the bar
        ha = 'right' if invert_axis else 'left'

        # Add text inside the bar
        plt.text(
            x_position,
            y_position,
            label,
            ha=ha, 
            va='center',
            color='black' if amount > bar_width * 0.1 else 'white',  # Contrast text color depending on bar size
            fontsize=8
        )


def plot_top_intensities_old(data, domain, title, path_figs):
    minlat, maxlat, minlon, maxlon = domain
    filtered_data = data[(data['LATITUDE'] >= minlat) & (data['LATITUDE'] <= maxlat) &
                         (data['LONGITUDE'] >= minlon) & (data['LONGITUDE'] <= maxlon)]

    # Convert 'datetime' column to datetime type
    filtered_data['TIME_EVENT'] = pd.to_datetime(filtered_data['TIME_EVENT'])

    # Filter top 20 for each event type based on intensity
    top_precip = filtered_data[filtered_data['TYPE_EVENT'] == 'PRECIP'].nlargest(20, 'PRECIPITATION_AMOUNT').reset_index()
    top_hail = filtered_data[filtered_data['TYPE_EVENT'] == 'HAIL'].nlargest(20, 'MAX_HAIL_DIAMETER').reset_index()

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    # Plotting Precipitation
    precip_bars = ax1.barh(top_precip.index, top_precip['PRECIPITATION_AMOUNT'], color='#add8e6', alpha=0.6)
    ax1.set_xlabel('PRECIPITATION AMOUNT (mm)', fontsize=12)
    ax1.invert_xaxis()  # Invert x-axis for left plot
    ax1.set_yticks([])  # Remove y-axis ticks
    ax1.set_ylim(len(top_precip) - 0.5, -0.5) # Set the y-axis limits to invert them

    # Plotting Hail
    hail_bars = ax2.barh(top_hail.index, top_hail['MAX_HAIL_DIAMETER'], color='#ffa07a', alpha=0.6)
    ax2.set_xlabel('MAX HAIL DIAMETER (cm)', fontsize=12)
    ax2.set_yticks([])  # Remove y-axis ticks
    ax2.set_ylim(len(top_hail) - 0.5, -0.5) # Set the y-axis limits to invert them

    # Add labels inside the bars
    add_labels_inside(precip_bars, top_precip['PRECIPITATION_AMOUNT'], top_precip['TIME_EVENT'], top_precip['LATITUDE'], top_precip['LONGITUDE'], invert_axis=True)
    add_labels_inside(hail_bars, top_hail['MAX_HAIL_DIAMETER'], top_hail['TIME_EVENT'], top_hail['LATITUDE'], top_hail['LONGITUDE'])

    # Set titles
    ax1.set_title('Top 20 Precipitation Intensities', fontsize=14)
    ax2.set_title('Top 20 Hail Intensities', fontsize=14)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Adjust subplots and save plot
    plt.subplots_adjust(wspace=0.5, top=0.85)
    plt.tight_layout()
    fig.savefig(f'{path_figs}{title}.png', bbox_inches='tight')

    # Show plot
    plt.show()


def plot_top_intensities(data, domain, title, path_figs, path_raster):
    minlat, maxlat, minlon, maxlon = domain
    filtered_data = data[(data['LATITUDE'] >= minlat) & (data['LATITUDE'] <= maxlat) &
                         (data['LONGITUDE'] >= minlon) & (data['LONGITUDE'] <= maxlon)]

    # Convert 'datetime' column to datetime type
    filtered_data['TIME_EVENT'] = pd.to_datetime(filtered_data['TIME_EVENT'])

    # Filter top 20 for each event type based on intensity
    top_precip = filtered_data[filtered_data['TYPE_EVENT'] == 'PRECIP'].nlargest(20, 'PRECIPITATION_AMOUNT').reset_index()
    top_hail = filtered_data[filtered_data['TYPE_EVENT'] == 'HAIL'].nlargest(20, 'MAX_HAIL_DIAMETER').reset_index()

    # Create subplots using GridSpec
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])

    ax1 = plt.subplot(gs[0, 0])  # Top left
    ax2 = plt.subplot(gs[1, 0])  # Bottom left
    ax3 = plt.subplot(gs[:, 1], projection=ccrs.PlateCarree())  # Right side

    # Plotting Precipitation
    precip_bars = ax1.barh(top_precip.index, top_precip['PRECIPITATION_AMOUNT'], color='blue', alpha=0.6)
    ax1.set_xlabel('Precipitation Amount (mm)', fontsize=14)
    ax1.set_yticks(top_precip.index)
    ax1.set_yticklabels(top_precip['TIME_EVENT'].dt.strftime('%Y-%m-%d %H:%M'),fontsize=12)
    ax1.set_title('Top 20 Precipitation Intensities', fontsize=14)
    ax1.tick_params(axis='x', which='major', labelsize=12)
    ax1.set_ylim(len(top_precip) - 0.5, -0.5) # Set the y-axis limits to invert them


    # Plotting Hail
    hail_bars = ax2.barh(top_hail.index, top_hail['MAX_HAIL_DIAMETER'], color='red', alpha=0.6)
    ax2.set_xlabel('Max Hail Diameter (cm)', fontsize=14)
    ax2.set_yticks(top_hail.index)
    ax2.set_yticklabels(top_hail['TIME_EVENT'].dt.strftime('%Y-%m-%d %H:%M'), fontsize=12)
    ax2.set_title('Top 20 Hail Intensities', fontsize=14)
    ax2.tick_params(axis='x', which='major', labelsize=12)
    ax2.set_ylim(len(top_precip) - 0.5, -0.5) # Set the y-axis limits to invert them



    # Plot raster
    with rasterio.open(path_raster) as src:
        # Create a map plot
        ax3.set_extent([minlon, maxlon, minlat, maxlat])
        ax3.add_feature(cfeature.BORDERS, linestyle=':')
        ax3.add_feature(cfeature.COASTLINE)
        ax3.add_feature(cfeature.LAKES, alpha=0.5)
        ax3.add_feature(cfeature.RIVERS)

        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        ax3.imshow(src.read(1), origin='upper', cmap='gist_earth', extent=extent, transform=ccrs.PlateCarree(), alpha=0.5, interpolation='spline36')
        
        ax3.add_feature(cfeature.OCEAN, color='blue')

        # Add lat-lon axis tick labels
        gl = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 12, 'color': 'black'}
        gl.ylabel_style = {'size': 12, 'color': 'black'}

    # Plot top events with numbers on the map
    for i, row in top_precip.iterrows():
        ax3.text(row['LONGITUDE'], row['LATITUDE'], str(i+1), fontsize=12, ha='center', va='center',
                 color='blue', weight='bold', transform=ccrs.Geodetic())

    for i, row in top_hail.iterrows():
        ax3.text(row['LONGITUDE'], row['LATITUDE'], str(i+1), fontsize=12, ha='center', va='center',
                 color='red', weight='bold', transform=ccrs.Geodetic())

    # Adding a legend for the map
    blue_patch = mpatches.Patch(color='blue', label='Precipitation')
    red_patch = mpatches.Patch(color='red', label='Hail')
    ax3.legend(handles=[blue_patch, red_patch], loc='lower left', fontsize=12)

    # Adjust the title space and layout of the plots
    fig.suptitle(title, fontsize=16, fontweight='bold')  # Reduce the pad to bring the title closer to the plots
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, wspace=0.1, hspace=0.4)

    # Save plot
    fig.savefig(f'{path_figs}{title}.png', dpi=300, bbox_inches='tight')

    # Show plot
    plt.tight_layout()
    plt.show()

