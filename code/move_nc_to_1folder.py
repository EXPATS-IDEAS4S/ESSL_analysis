"""
this code is used to copy all nc files contained in the input directory
/sat_data/crops/GRL_testing_crops and related subfolders 
to the directory dir_path+"/1/" to have the dataset ready 
for the run of VISSL test 

author: Claudia Acquistapace
date: 2026-06-10

how to run:
conda activate venv_vissl

set the source and the destination path for your VISSL configuration in the main function of this script, then run:
python move_nc_to_1folder.py

"""

import os
import shutil


def main():

    # Define the source directory containing the .nc files
    target_dir = os.path.join(dir_path, "1")

    # create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Walk through the source directory and its subdirectories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".nc"):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)

                # Copy the file to the target directory, keeping the original file in place
                shutil.copy2(source_file, target_file)
                print(f"Copied: {source_file} to {target_file}")


if __name__ == "__main__":

    # define source and target directories
    source_dir = "/sat_data/crops/GRL_testing_crops"
    dir_path = "/sat_data/crops/test_grl_2026"
    main()  
