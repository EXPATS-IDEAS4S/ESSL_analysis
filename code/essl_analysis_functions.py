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


def plot_events_paula(datasets, main_domain, subdomains, intensity, raster_path):

    # Extract main domain boundaries
    minlat_main, maxlat_main, minlon_main, maxlon_main = main_domain

    with rasterio.open(raster_path) as src:
        # Create a map plot
        fig, ax = plt.subplots(figsize=(12, 8),subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([minlon_main, maxlon_main, minlat_main, maxlat_main])
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        ax.imshow(src.read(1), origin='upper', cmap='gist_earth', extent=extent, transform=ccrs.PlateCarree(), alpha=0.5)

        ax.add_feature(cfeature.OCEAN, color='blue')

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

            if intensity:
                sc1 = ax.scatter(rain_data['lon'], rain_data['lat'], 
                        c=rain_data['mm'], marker='o', label=label_rain, 
                        cmap='Blues', transform=ccrs.PlateCarree())
                
                sc2 = ax.scatter(hail_data['lon'], hail_data['lat'], 
                        c=hail_data['size (cm)'], marker='^', label=label_hail, 
                        cmap='Reds', transform=ccrs.PlateCarree())
            else:
                ax.scatter(rain_data['lon'], rain_data['lat'], 
                        marker='o', label=label_rain, c='Blue',
                        transform=ccrs.PlateCarree())
                
                ax.scatter(hail_data['lon'], hail_data['lat'], 
                        marker='^', label=label_hail, c='Red',
                        transform=ccrs.PlateCarree())
        

        # Adjust legend location
        ax.legend(loc='upper left')#, bbox_to_anchor=(1, 0.5))

        # Show colorbars at the bottom if intensity is True
        if intensity:
            cax1 = fig.add_axes([0.15, 0.08, 0.35, 0.02])
            cax2 = fig.add_axes([0.55, 0.08, 0.35, 0.02])
            
            fig.colorbar(sc1, cax=cax1, orientation='horizontal', label='Precipitation Amount (mm)')
            fig.colorbar(sc2, cax=cax2, orientation='horizontal', label='Hail Diameter (cm)')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()