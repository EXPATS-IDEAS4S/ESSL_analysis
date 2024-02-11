'''
reader functions for csv files from essl database
'''

#import libraries
import pandas as pd
import numpy as np
import os


def read_csv(filename, path_name):
    """
    function to read csv files

    Args:
        filename (string): name of the file
        path_name (string): string indicating input path

    Returns:
        _type_: _description_
    """
    # read csv file
    data = pd.read_csv(path_name+filename, low_memory=False)
    
    return data


