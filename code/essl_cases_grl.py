"""""

This script reads the file /sat_data/output/grl/csv/eswd-v2-2012-2025_expats.csv and derives a list of case studies
for the years 2025 to be used for test in the grl2026 publication, based on the criteria that:
- the cases should be from the year 2025
- the cases should have generated a large number of reports (>thr_reports reports in a day)
- the cases should be localized in the po valley mainly ( not really posed as a condition a the moment,
 but we can check the lat and lon of the reports to select only those cases that are localized in the po valley)
- it classifies the case based on the number and type of reports (rain, hail)

Once selected the cases, the code stores the data as rows of video sequences:
- data, start time, end time, start_event, end_event, lat, lon, number of reports, number of rain reports, number of hail reports,
 duration of the case, case type 

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

# years that can be used as test
test_years = [2016, 2017, 2020, 2021, 2024, 2025]


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

# loop on test years to select the test dataset

for test_year in test_years:

    print(f"Number of reports in {test_year}: {eswd_df[eswd_df['year'] == test_year].shape[0]}")

    # group by date and count number of reports for each day
    eswd_test_year_df = eswd_df[eswd_df["year"] == test_year]
    reports_per_day = eswd_test_year_df.groupby("date").size().reset_index(name="num_reports")


    # print days of the year with the highest number of reports
    print(f"Dates in {test_year} with the highest number of reports:")
    print(reports_per_day.sort_values(by="num_reports", ascending=False).head(50))   


    # for each date, read the column "TYPE_EVENT" and count the number of reports with
    # type PRECIP and the number of reports with type HAIL and assign 
    # precip only if more than 80% of the reports are PRECIP
    # selecting cases with more than 50 reports in a day
    thr_reports = 20
    case_studies = []
    for index, row in reports_per_day.iterrows():
        date = row["date"]
        num_reports = row["num_reports"]
        if num_reports > thr_reports:
            day_reports = eswd_test_year_df[eswd_test_year_df["date"] == date]
            num_precip = day_reports[day_reports["TYPE_EVENT"] == "PRECIP"].shape[0]
            num_hail = day_reports[day_reports["TYPE_EVENT"] == "HAIL"].shape[0]
            if num_precip / num_reports > 0.8:
                case_type = "PRECIP"
            else:
                case_type = "HAIL"
            case_studies.append({
                "date": date,
                "num_reports": num_reports,
                "num_precip": num_precip,
                "num_hail": num_hail,
                "case_type": case_type
            })

    # establish start and end time as the first time stamp and last time stamp of the day in which the reports were collected
    for case in case_studies:
        
        date = case["date"]
        day_reports = eswd_test_year_df[eswd_test_year_df["date"] == date]
        # sort day reports by time event
        day_reports = day_reports.sort_values(by="TIME_EVENT")

        # store start and end time of the event (first and last report of the day) in the case dictionary
        start_event = day_reports["TIME_EVENT"].iloc[0] 
        end_event = day_reports["TIME_EVENT"].iloc[-1]

        # check if end_event is on the next day then print a warning message
        if pd.to_datetime(end_event).date() > pd.to_datetime(start_event).date():
            print(f"Warning: end_event {end_event} is on the next day after start_event {start_event} for date {date}")


        # build time stamp in the format yyyy-mm=ddT00:00:00.000Z and yyyy-mm-ddT23:59:59.999Z
        start_time = f"{date.strftime('%Y-%m-%d')}T00:00:00.000Z"
        end_time = f"{date.strftime('%Y-%m-%d')}T23:59:59.999Z"

        case["start_time"] = start_time
        case["end_time"] = end_time
        case["start_event"] = start_event
        case["end_event"] = end_event



        # calculate duration of the event in hours
        start_time_dt = pd.to_datetime(start_event)
        end_time_dt = pd.to_datetime(end_event)
        duration_hours = (end_time_dt - start_time_dt).total_seconds() / 3600
        case["duration_hours"] = duration_hours

        # calculate start and end lat and lon by reading the lat and lon of the first and last report of the day 
        start_lat = day_reports["LATITUDE"].iloc[0]
        start_lon = day_reports["LONGITUDE"].iloc[0]
        end_lat = day_reports["LATITUDE"].iloc[-1]
        end_lon = day_reports["LONGITUDE"].iloc[-1]
        case["start_lat"] = start_lat
        case["start_lon"] = start_lon
        case["end_lat"] = end_lat
        case["end_lon"] = end_lon


    # write all days with more than thr reports in a csv file, with lines containing
    # data, start time, end time, lat, lon, number of reports, number of rain reports, number of hail reports, duration of the case, case type 
    output_filename = f"essl_cases_{test_year}_grl.csv"
    case_studies_df = pd.DataFrame(case_studies)

    # sort cases by number of reports
    case_studies_df = case_studies_df.sort_values(by="num_reports", ascending=False)

    case_studies_df.to_csv(os.path.join(output_dir, output_filename), index=False)
    print(f"Case studies saved to {output_dir+output_filename}")
