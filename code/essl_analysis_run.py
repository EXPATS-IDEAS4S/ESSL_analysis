"""
Analysis of the ESSL database 
about the extreme weather events
over the EXPATS and TeamX domain
"""

#import libraries
import pandas as pd
import numpy as np

import essl_analysis_functions

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Disable the limit

# define paths
path_file = '/home/daniele/Documenti/PhD_Cologne/TeamX/data/'

#define filenames
filename_essl = path_file+'ESWD_HAIL_PRECIP_TORNADO_WIND_5-16_42-51_5_20230101-20231010_v1_6.csv'
filename_paula_red = path_file+'Red_domain_lon11.00-11.81_lat45.40-46.95.csv'
filename_paula_grey = path_file+'Grey_domain_lon10.7-11.5_lat45.6-46.6.csv'

# open csv files
data_essl = pd.read_csv(filename_essl, low_memory=False)
data_paula_red = pd.read_csv(filename_paula_red)
data_paula_grey = pd.read_csv(filename_paula_grey)

# define domain of interest # [minlat, maxlat, minlon, maxlon]
domain_grey = [45.6, 46.6, 10.7, 11.5] #TeamX domain (Grey domain)
domain_red = [46.40, 46.95, 11, 11.81] #TeamX domain (Red domain) NB: minlat is 46.40 and not 45.40 as reported in filename
domain_expats = [42, 51.5, 5, 16] #EXPATS domain

# Padding value
padding = 0.2

# Determine encompassing domain
left_border = min(domain_grey[0], domain_red[0]) - padding
right_border = max(domain_grey[1], domain_red[1]) + padding
bottom_border = min(domain_grey[2], domain_red[2]) - padding
top_border = max(domain_grey[3], domain_red[3]) + padding

encompassing_domain = [left_border, right_border, bottom_border, top_border]

# plot the spatia distribution of the events
#essl_analysis_functions.plot_events_essl(data_essl, domain_large, False)
#essl_analysis_functions.plot_events_paula([data_paula_red,data_paula_grey], domain_expats, [domain_red,domain_grey], False, raster_file_path)
raster_filename = 'NE1_HR_LC_SR_W_DR/NE1_HR_LC_SR_W_DR.tif'
title = 'ESSL Events Location 2013-2019'
essl_analysis_functions.plot_events_paula([data_paula_red,data_paula_grey], encompassing_domain, [domain_red,domain_grey], False, raster_filename, title, path_file)


#TODO add orography and plot the entire domain of EXPAT with highlith on the TEAMX domain
#TODO legend outside and adjust colorbars

# TODO plot the temporal trend of the events, monthly frequency and (gif with events evoluton)?