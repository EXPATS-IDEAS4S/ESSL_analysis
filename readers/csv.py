'''
reader functions for csv files from essl database
'''

#import libraries
import pandas as pd
import numpy as np
import os
import glob


def read_csv(filename, path_name):
    """
    function to read csv files

    Args:
        filename (string): name of the file
        path_name (string): string indicating input path

    Returns:
        data: pandas dataframe containing the data
    """
    # read csv file
    data = pd.read_csv(path_name+filename, low_memory=False)
    
    return data


def read_csv_all():
    """
    function to read and merge all data from ESSL
    
    Returns:
        data: pandas dataframe containing the data
    """
    # reading input datasets
    data_2010_2020 = pd.read_csv('/net/yube/ESSL/ESSL_Events_2010_2020.csv', low_memory=False)
    data_2021_2023 = pd.read_csv('/net/yube/ESSL/Prec_Hail_ExpatsDomain_21-23.csv', low_memory=False)

    # merging data 
    data = pd.concat([data_2010_2020, data_2021_2023])
    
    return data
