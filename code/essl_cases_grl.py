"""""

This script reads the file /sat_data/output?grl?csv/eswd-v2-2012-2025_expats.csv and derives a list of case studies
for the years 2025 based on the criteria that:
- the cases should be from the year 2025
- the cases should have generated a large number of reports
- the cases should be localized in the po valley mainly
- it classifies the case based on the number and type of reports (rain, hail)

Once selected the cases, the code stores the data as rows of video sequences:
- data, start time, end time, lat, lon, number of reports, number of rain reports, number of hail reports, duration of the case, case type 

The output csv file is stored in /sat_data/output/grl/csv/essl_cases_2025_grl.csv

author: claudia Acquistapace
date: 2026-06-09


"""


import os
import sys
import numpy as np
import pandas as pd
from glob import glob
import sys
import pdb


# === IMPORT HELPER FUNCTIONS ===
sys.path.append("/home/claudia/codes/ML_postprocessing")
from utils.processing.features_utils import load_tsne_coordinates
from utils.plotting.class_colors import colors_per_class1_names
from utils.configs import load_config

# read csv file containing the eswd reports for the years 2012-2025
input_dir = "/sat_data/output/grl_2026/csv/"
csv_filename = "eswd-v2-2012-2025_expats.csv"
eswd_df = pd.read_csv(os.path.join(input_dir, csv_filename))
print("ESWD CSV file loaded successfully.")
output_dir = input_dir

# read column of time event
time_event = eswd_df["TIME_EVENT"]

# extract years from a time stamp string of the format 2025-09-30T19:10:47.702Z
years = time_event.str.slice(0, 4).astype(int)
months = time_event.str.slice(5, 7).astype(int)
days = time_event.str.slice(8, 10).astype(int)
hours = time_event.str.slice(11, 13).astype(int)
minutes = time_event.str.slice(14, 16).astype(int)
seconds = time_event.str.slice(17, 19).astype(int)


# create a new columns in the dataframe with the extracted years
eswd_df["year"] = years
eswd_df["month"] = months
eswd_df["day"] = days
eswd_df["hour"] = hours
eswd_df["minute"] = minutes
eswd_df["second"] = seconds
eswd_df["date"] = pd.to_datetime(eswd_df[["year", "month", "day"]])

# group by date and count number of reports for each day
eswd_2025_df = eswd_df[eswd_df["year"] == 2025]
reports_per_day = eswd_2025_df.groupby("date").size().reset_index(name="num_reports")


# print days of the year with the highest number of reports
print("Dates in 2025 with the highest number of reports:")
print(reports_per_day.sort_values(by="num_reports", ascending=False).head(50))   


# for each date, read the column "TYPE_EVENT" and count the number of reports with type PRECIP and the number of reports with type HAIL and assign 
# precip only if more than 80% of the reports are PRECIP
case_studies = []
for index, row in reports_per_day.iterrows():
    date = row["date"]
    num_reports = row["num_reports"]
    if num_reports > 80:
        day_reports = eswd_2025_df[eswd_2025_df["date"] == date]
        num_precip = day_reports[day_reports["TYPE_EVENT"] == "PRECIP"].shape[0]
        num_hail = day_reports[day_reports["TYPE_EVENT"] == "HAIL"].shape[0]
        if num_precip / num_reports > 0.8:
            case_type = "PRECIP"
        elif num_hail / num_reports > 0.8:
            case_type = "HAIL"
        else:
            case_type = "MIXED"
        case_studies.append({
            "date": date,
            "num_reports": num_reports,
            "num_precip": num_precip,
            "num_hail": num_hail,
            "case_type": case_type
        })

# establish start and end time by reading first and last report of the day
for case in case_studies:
    
    date = case["date"]
    day_reports = eswd_2025_df[eswd_2025_df["date"] == date]
    start_time = day_reports["TIME_EVENT"].min()
    end_time = day_reports["TIME_EVENT"].max()
    case["start_time"] = start_time
    case["end_time"] = end_time

    # calculate duration of the case in hours
    start_time_dt = pd.to_datetime(start_time)
    end_time_dt = pd.to_datetime(end_time)
    duration_hours = (end_time_dt - start_time_dt).total_seconds() / 3600
    case["duration_hours"] = duration_hours


# write all days with more than 80 reports in a csv file, with lines containing
# data, start time, end time, lat, lon, number of reports, number of rain reports, number of hail reports, duration of the case, case type 
output_filename = "essl_cases_2025_grl.csv"
case_studies_df = pd.DataFrame(case_studies)
case_studies_df.to_csv(os.path.join(output_dir, output_filename), index=False)
print(f"Case studies saved to {output_filename}")
