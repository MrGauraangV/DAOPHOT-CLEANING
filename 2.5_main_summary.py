import pandas as pd
import glob
import os

root_dir = 'star_files_1'

files = [os.path.join(root_dir, f) for f in glob.glob('*.dat', root_dir=root_dir)]

summary_rows = []

for path in files:
    df = pd.read_csv(path, delimiter='\s+')
    df = df[['NX','NY']]
    summary_rows.append({
        "Star": path,
        "X_mean": df["NX"].mean(),
        "Y_mean": df["NY"].mean(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.round(3)

summary_df.to_csv('Raw Summary.dat', index=False)