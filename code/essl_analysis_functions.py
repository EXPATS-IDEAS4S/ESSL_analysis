"""
plotting function used
to analyze ESSL data
"""

#import libraries
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
import cartopy.io.shapereader as shpreader



def plot_events_essl(data, domain, intensity):    

    # Filter data based on the provided domain [minlat, maxlat, minlon, maxlon]
    minlat, maxlat, minlon, maxlon = domain
    filtered_data = data[(data['LATITUDE']/1000 >= minlat) & (data['LATITUDE']/1000 <= maxlat) &
                         (data['LONGITUDE']/1000 >= minlon) & (data['LONGITUDE']/1000 <= maxlon)]

    # Create a map plot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([minlon, maxlon, minlat, maxlat])
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')

    # get rain events
    rain_data = filtered_data[filtered_data['TYPE_EVENT'] == 'PRECIP']
    print(rain_data)

    # get hail events
    hail_data = filtered_data[filtered_data['TYPE_EVENT'] == 'HAIL']
    print(hail_data)
    
    
    if intensity:
        sc1 = ax.scatter(rain_data['LONGITUDE']/1000, rain_data['LATITUDE']/1000, 
                     c=rain_data['PRECIPITATION_AMOUNT'], marker='o', label='Rain', 
                     cmap='Blues', transform=ccrs.PlateCarree())
        sc2 = ax.scatter(hail_data['LONGITUDE']/1000, hail_data['LATITUDE']/1000, 
                     c=hail_data['AVERAGE_HAIL_DIAMETER'], marker='^', label='Hail', 
                     cmap='Reds', transform=ccrs.PlateCarree())
        # Colorbar 
        plt.colorbar(sc1, ax=ax, label='PRECIPITATION_AMOUNT (mm)')
        plt.colorbar(sc2, ax=ax, label='AVERAGE_HAIL_DIAMETER (cm)')
    else:
        sc1 = ax.scatter(rain_data['LONGITUDE']/1000, rain_data['LATITUDE']/1000, 
                     c='Blue', marker='o', label='Rain', 
                     transform=ccrs.PlateCarree())
        sc2 = ax.scatter(hail_data['LONGITUDE']/1000, hail_data['LATITUDE']/1000, 
                     c='Red', marker='^', label='Hail', 
                    transform=ccrs.PlateCarree())
    #set legend
    ax.legend()

    # Show the plot
    plt.show()


def plot_events_paula(datasets, main_domain, subdomains, intensity, raster_filename, title, path_file, path_figs):

    # Extract main domain boundaries
    minlat_main, maxlat_main, minlon_main, maxlon_main = main_domain

    with rasterio.open(path_file+raster_filename) as src:
        # Create a map plot
        fig, ax = plt.subplots(figsize=(12, 8),subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([minlon_main, maxlon_main, minlat_main, maxlat_main])
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAKES)
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