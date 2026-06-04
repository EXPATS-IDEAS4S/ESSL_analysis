# ESSL Analysis

Scripts for analysing the ESSL extreme-weather event database and generating storm-crop datasets for plotting, tracking, clustering, and daily-event visualisation.

## Main scripts

- `essl_analysis_run.py`: example analysis workflow for the ESSL event database.
- `essl_analysis_functions.py`: plotting and summary helpers for event distributions and trends.
- `cluster_events.py` and `data_utils.py`: clustering and spatial grouping utilities.
- `storm_trajectory_tracking.py` and `check_trajectories.py`: trajectory analysis helpers.
- `crop_generator/`: scripts for building crops from NetCDF files or S3 buckets.
- `analyse_crops/`: plotting and GIF helpers for crop inspection.

## Setup

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

- Many scripts use local hard-coded paths for datasets and output folders, so update those paths before running.
- Some crop-generation scripts expect a local credentials file such as `credentials_buckets.py`, which is ignored by Git.
- The project does not currently include a single unified CLI; most files are standalone scripts.
