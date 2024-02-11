"""
domain definitions
"""

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

# raster filename
raster_filename = 'NE1_HR_LC_SR_W_DR/NE1_HR_LC_SR_W_DR.tif'
