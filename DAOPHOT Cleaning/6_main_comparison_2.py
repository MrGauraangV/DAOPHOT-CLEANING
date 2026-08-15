import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from itertools import combinations

source_dir = "ref_summary.csv"
data_dir = "star_files_2"

file_paths = [os.path.join(data_dir, f) for f in glob.glob("*.dat", root_dir=data_dir)]
sdf = pd.read_csv(source_dir, delimiter=',')

def locate(df, column, i):
    return df.loc[df[column] == i]

master_data = [pd.read_csv(file) for file in file_paths]
master_df = pd.concat(master_data).sort_values(by='Star')

final_results = []
for i in range(len(file_paths)+1):

    source_located = locate(sdf, 'focus_star', i)
    list_of_refs = np.array(source_located['comparison_star'])

    if len(list_of_refs) == 0:
        final_results.append({
            'focus_star': i,
            'ref_1': None,
            'ref_2': None,
            'std': None
        })
        print("Completed:", i, "- No reference stars.")
        continue

    if len(list_of_refs) == 1:
        final_results.append({
            'focus_star': i,
            'ref_1': list_of_refs[0],
            'ref_2': None,
            'std': None
        })
        print("Completed:", i, "- Only one reference star.")
        continue

    best_pair = None
    best_std = float('inf')

    for ref_star_1, ref_star_2 in combinations(list_of_refs, 2):

        master_locate_1 = locate(master_df, 'Star', ref_star_1)
        master_locate_2 = locate(master_df, 'Star', ref_star_2)

        merged = pd.merge(master_locate_1, master_locate_2, how='inner', on=['JD'])

        if len(merged) == 0:
            continue

        mag_diff = abs(merged['Magnitude_x'] - merged['Magnitude_y'])
        std_val = mag_diff.std()

        if std_val < best_std:
            best_std = std_val
            best_pair = (ref_star_1, ref_star_2)

    # If all pairs produced empty merges:
    if best_pair is None:
        final_results.append({
            'focus_star': i,
            'ref_1': None,
            'ref_2': None,
            'std': None
        })
    else:
        final_results.append({
            'focus_star': i,
            'ref_1': best_pair[0],
            'ref_2': best_pair[1],
            'std': best_std
        })

    print("Completed:", i)

final_results_df = pd.DataFrame(final_results)
final_results_df.to_csv("Reference Stars.csv", index=False)
