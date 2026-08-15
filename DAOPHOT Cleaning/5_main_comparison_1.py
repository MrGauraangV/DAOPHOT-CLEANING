import pandas as pd
import numpy as np
import os

input_dir = 'Summary of Stars.dat'
out_dir = 'ref_summary.csv'

summary = pd.read_csv(input_dir, delimiter=',').reset_index(drop=True)

PIXEL_TOL = 100
summary_1 = []

for focus_star in range(1, len(summary)+1):

    fs_mag = summary.loc[focus_star - 1, "Mag_mean"]
    fs_x   = summary.loc[focus_star - 1, "X_mean"]
    fs_y   = summary.loc[focus_star - 1, "Y_mean"]

    for index, row in summary.iterrows():
        x_diff = fs_x - row["X_mean"]
        y_diff = fs_y - row["Y_mean"]

        pixel_distance = np.sqrt(x_diff**2 + y_diff**2)

        summary_1.append({
            "focus_star": focus_star,
            "comparison_star": row["Star"],
            "avg_mag_diff": abs(fs_mag - row["Mag_mean"]),
            "pixel_distance": pixel_distance
        })

df = pd.DataFrame(summary_1)

df_filtered = df[
        df['avg_mag_diff'].between(0, 1, inclusive='right') &
        (df['pixel_distance'] < PIXEL_TOL)
    ]

df_filtered.round(3).to_csv(out_dir, index=False)
