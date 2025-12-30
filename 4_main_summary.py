import pandas as pd
import glob
import os

root_dir = 'star_files_2'

files = [os.path.join(root_dir, f) for f in glob.glob('*.dat', root_dir=root_dir)]
dfs = [pd.read_csv(path, delimiter=',') for path in files]

summary_rows = []

for df in dfs:

    # Extract the TRUE star ID inside the file
    star_id = df["Star"].iloc[0]

    summary_rows.append({
        "Star": star_id,
        "X_mean": df["NX"].mean(),
        "Y_mean": df["NY"].mean(),
        "Mag_mean": df["Magnitude"].mean(),
        "Mag_std": df["Magnitude"].std(),
        "Err_mean": df["Mag_Err"].mean(),
        "Err_std": df["Mag_Err"].std(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.round(3)

summary_df.sort_values(by='Star').to_csv('Summary of Stars.dat', index=False)