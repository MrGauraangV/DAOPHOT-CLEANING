import pandas as pd
import glob
import os
import numpy as np

root_directory = "clean_files"
output_dir = "star_files_1_new"

os.makedirs(output_dir, exist_ok=True)

file_paths = [os.path.join(root_directory, f) 
              for f in glob.glob("*.csv", root_dir=root_directory)]

print(f"Found {len(file_paths)} CSV files.")

df_list = []
for idx, path in enumerate(file_paths, start=1):

    try:
        df_list.append(pd.read_csv(path))
    except Exception as e:
        print(f"Error reading {path}: {e}")
        continue

    if idx % 500 == 0:
        print(f"Loaded {idx}/{len(file_paths)} files...")

print("Concatenating all data into DataFrame...")
df = pd.concat(df_list, ignore_index=True)
del df_list 

print("All CSV files loaded.")
print("Rows in DataFrame:", len(df))

PIXEL_TOL = 1.5
PIXEL_TOL2 = PIXEL_TOL ** 2

df = df.sort_values(by=["NX", "NY"]).reset_index(drop=True)

clusters = []
visited = np.zeros(len(df), dtype=bool)

print("Starting clustering...\n")

for i in range(len(df)):
    if visited[i]:
        continue

    base_x = df.loc[i, "NX"]
    base_y = df.loc[i, "NY"]

    print(f"[Cluster {len(clusters)+1}] Base star NX={base_x:.2f}, NY={base_y:.2f} (row {i})")

    cluster = [i]
    visited[i] = True

    for j in range(i + 1, len(df)):
        if visited[j]:
            continue

        dx = df.loc[j, "NX"] - base_x

        if dx > PIXEL_TOL:
            break

        dy = df.loc[j, "NY"] - base_y
        dist2 = dx*dx + dy*dy

        if dist2 <= PIXEL_TOL2:
            cluster.append(j)
            visited[j] = True

    clusters.append(cluster)

print("\nClustering complete.")
print(f"Total clusters found: {len(clusters)}")

print("\nSaving clusters...")
for idx, c in enumerate(clusters, start=1):
    star_df = df.iloc[c].sort_values(by="JD")
    file_path = os.path.join(output_dir, f"Star_{idx}_Photometry.dat")
    star_df.to_csv(file_path, sep="\t", index=False)

print("Done. All clusters saved.")
