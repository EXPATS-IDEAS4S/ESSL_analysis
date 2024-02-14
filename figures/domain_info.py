"""
domain definitions
"""

# define domain of interest # [minlat, maxlat, minlon, maxlon]
domain_grey = [45.6, 46.6, 10.7, 11.5] #TeamX domain (Grey domain)
domain_red = [46.40, 46.95, 11, 11.81] #TeamX domain (Red domain) NB: minlat is 46.40 and not 45.40 as reported in filename
domain_expats = [42, 51.5, 5, 16] #EXPATS domain
#domain_dfg = [45.267, 47.569, 10.14, 12.614]  # dfg subdomain
domain_dfg =  [45.5, 47., 10.75, 11.75]
#domain_dfg =  [46., 46.75, 11, 11.75] # supersmall, for our site details

# Padding value
padding = 0.2

# Determine encompassing domain
def det_domain(domain, padding=0.):
    """
    determine encompassing domain based on padding

    Args:
        domain (list): list of lat/lon values
        padding (scalar): padding scalar value
    """
    
    left_border = domain[0] - padding
    right_border = domain[1] + padding
    bottom_border = domain[2] - padding
    top_border = domain[3] + padding
    domain_final = [left_border, right_border, bottom_border, top_border]

    return(domain_final)


# raster filename
raster_filename = 'NE1_HR_LC_SR_W_DR/NE1_HR_LC_SR_W_DR.tif'
