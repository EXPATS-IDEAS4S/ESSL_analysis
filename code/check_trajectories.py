import pandas as pd
import os

from data_utils import haversine_km
from plot_utils import trajectory_diagnostics_plot

path_trajectories = "/work/dcorradi/ESSL/1st_paper/grouped_output_15min_eps150km/trajectories/storm_trajectories.csv"
path_grouped = "/work/dcorradi/ESSL/1st_paper/grouped_output_15min_eps150km/events_grouped.csv"
path_summary = "/work/dcorradi/ESSL/1st_paper/grouped_output_15min_eps150km/events_summary.csv"

df_traj = pd.read_csv(path_trajectories)
df_grouped = pd.read_csv(path_grouped)
df_summary = pd.read_csv(path_summary)

output_dir = "/work/dcorradi/ESSL/1st_paper/grouped_output_15min_eps150km/trajectories/"
os.makedirs(output_dir, exist_ok=True)

print("Trajectories:")
print(df_traj.head())
# print("\nGrouped events:")
# print(df_grouped.head())
# print("\nSummary:")
# print(df_summary.head())

#print some statistics of the trajectories
n_trajectories = df_traj["storm_id"].nunique()
print(f"\nNumber of unique storm trajectories: {n_trajectories}")
lifetime_stats = df_traj.groupby("storm_id").size().describe()
print("\nStorm lifetime statistics (in number of events):")
print(lifetime_stats)
#compute number of trajectories per storm type
type_counts = df_traj.groupby("storm_type")["storm_id"].nunique()
print("\nNumber of unique storm trajectories per event type:")
print(type_counts)
#compute number of trajectories that had type transitions
type_transition_counts = df_traj.groupby("storm_id")["type_transition"].max().values.sum()
print("\nNumber of unique storm trajectories with type transitions:")
print(type_transition_counts)
#compute average distance traveled by each trajectory

def compute_trajectory_distance(traj):
    traj = traj.sort_values("time")
    total_distance = 0.0
    for i in range(1, len(traj)):
        lat1, lon1 = traj.iloc[i-1][["lat", "lon"]]
        lat2, lon2 = traj.iloc[i][["lat", "lon"]]
        distance = haversine_km(lat1, lon1, lat2, lon2)
        total_distance += distance
    return total_distance   

distance_stats = df_traj.groupby("storm_id").apply(compute_trajectory_distance).describe()
print("\nStorm trajectory distance statistics (in km):")
print(distance_stats)

#statistics on interplotated, exptrapolated observed points
n_total_points = len(df_traj)
n_interpolated_points = len(df_traj[df_traj["source"] == "interpolated"])
n_extrapolated_points = len(df_traj[df_traj["source"] == "extrapolated"])
print(f"\nTotal number of trajectory points: {n_total_points}")
print(f"Number of interpolated trajectory points: {n_interpolated_points} ({n_interpolated_points / n_total_points * 100:.2f}%)")
print(f"Number of extrapolated trajectory points: {n_extrapolated_points} ({n_extrapolated_points / n_total_points * 100:.2f}%)")


trajectory_diagnostics_plot(
    df_traj,
    n_days=10,
    max_slots=20,
    domain=(5, 16, 42, 51.5),
    res_deg=0.04,
    crop_pixels=100,
    seed=42,
    output_dir=output_dir,
)