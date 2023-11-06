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
#path_file = '/home/dcorradi/Documents/ESSL/Datasets/'
path_figs = '/home/daniele/Documenti/PhD_Cologne/TeamX/figs/'
#path_figs = 'home/dcorradi/Documents/ESSL/Figs/'

#define filenames
filename_essl = path_file+'ESWD_HAIL_PRECIP_TORNADO_WIND_5-16_42-51_5_20230101-20231010_v1_6.csv'
filename_paula_red = path_file+'Red_domain_lon11.00-11.81_lat45.40-46.95.csv'
filename_paula_grey = path_file+'Grey_domain_lon10.7-11.5_lat45.6-46.6.csv'
filename = path_file+'Prec_Hail_ExpatsDomain_21-22.csv'

# open csv files
data_essl = pd.read_csv(filename_essl, low_memory=False)
data_paula_red = pd.read_csv(filename_paula_red)
data_paula_grey = pd.read_csv(filename_paula_grey)
data = pd.read_csv(filename,low_memory=False)

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

teamx_domain = [left_border, right_border, bottom_border, top_border]

raster_filename = 'NE1_HR_LC_SR_W_DR/NE1_HR_LC_SR_W_DR.tif'
time_period = '2021-2022'

#plot the spatial map of the events

title = 'ESSL Events Location'
#essl_analysis_functions.plot_events_paula([data_paula_red,data_paula_grey], domain_expats, [domain_red,domain_grey], False, raster_file_path)
#essl_analysis_functions.plot_events_paula([data_paula_red,data_paula_grey], teamx_domain, [domain_red,domain_grey], False, raster_filename, title, path_file, path_figs)
#essl_analysis_functions.plot_events_essl(data_essl, teamx_domain)

#essl_analysis_functions.plot_events([data,data], teamx_domain, [domain_red,domain_grey], raster_filename, title+': '+time_period+' - '+'TeamX Domain', path_file, path_figs,True)
#essl_analysis_functions.plot_events([data], domain_expats, [domain_expats], raster_filename, title+': '+time_period+' - '+'Expats Domain', path_file, path_figs, False)


#plot the monthly frequency of the events

#title = 'Monthly Frequency of ESSL Events'
#essl_analysis_functions.plot_monthly_event_frequency(data,domain_expats,title+': '+time_period+' - '+'Expats Domain', path_figs)
#essl_analysis_functions.plot_monthly_event_frequency(data,teamx_domain,title+': '+time_period+' - '+'TeamX Domain' , path_figs)

# plot the temporal trend of the events with weekly resolution (TODO adjust the daily trend)
#title='Weekly Occurences Trends of ESSL Events'
#essl_analysis_functions.plot_event_trend(data,domain_expats,title+': '+time_period+' - '+'Expats Domain', path_figs, ['2021-01-01','2022-12-31'],'W')
#essl_analysis_functions.plot_event_trend(data,teamx_domain,title+': '+time_period+' - '+'TeamX Domain' , path_figs, ['2021-01-01','2022-12-31'],'W')

#title='Weekly Intensity Trends of ESSL Events'
#essl_analysis_functions.plot_intensity_trend(data,domain_expats,title+': '+time_period+' - '+'Expats Domain', path_figs, ['2021-01-01','2022-12-31'],'W')
#essl_analysis_functions.plot_intensity_trend(data,teamx_domain,title+': '+time_period+' - '+'TeamX Domain' , path_figs, ['2021-01-01','2022-12-31'],'W')

#plot events rankings
#title = 'ESSL Events Intensity Ranking'
#essl_analysis_functions.plot_top_intensities(data, domain_expats,title+': '+time_period+' - '+'Expats Domain', path_figs, path_file+raster_filename)
#essl_analysis_functions.plot_top_intensities(data, teamx_domain,title+': '+time_period+' - '+'TeamX Domain' , path_figs, path_file+raster_filename)


# plot daily events maps
title = 'ESSL Daily Events Map'
years = [2021,2022] 
for year in years:
    output_dir = '/home/daniele/Documenti/PhD_Cologne/TeamX/figs/Daily_cases/'+str(year)+'/'
    essl_analysis_functions.save_daily_event_maps(data, domain_expats, title+': Expats Domain', path_file+raster_filename, output_dir, year)
    essl_analysis_functions.save_daily_event_maps(data, teamx_domain, title+': TeamX Domain', path_file+raster_filename, output_dir, year)

# TODO gif with daily events evoluton (then pick few days and make hour resolutuion)