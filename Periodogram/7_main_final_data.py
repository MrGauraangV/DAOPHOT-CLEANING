import pandas as pd
import numpy as np
import os
import glob

ref_df = pd.read_csv("Reference Stars.csv", delimiter=',')

data_dir = "star_files_2"
output_dir = "star_files_3_dat"

file_paths = sorted(
    [os.path.join(data_dir, f) for f in glob.glob("*.dat", root_dir=data_dir)]
)

master_data = [pd.read_csv(file, delimiter=',') for file in file_paths]
master_df = pd.concat(master_data, ignore_index=True)

star_ids = sorted(master_df["Star"].unique())

def get_star_data(star_id):
    return master_df.loc[master_df["Star"] == star_id]

for star_id in star_ids:

    row = ref_df.loc[ref_df["focus_star"] == star_id]
    ref_star = row["ref_1"].values[0] if not row.empty else np.nan

    star_df = get_star_data(star_id)
    ref_df_data = get_star_data(ref_star) if not pd.isna(ref_star) else pd.DataFrame()

    if ref_df_data.empty:
        merged = star_df.copy()
    else:
        merged = pd.merge(star_df, ref_df_data, how="inner", on="JD")

    if merged.empty:
        merged = star_df.copy()

    have_mag_cols = ("Magnitude_x" in merged.columns) and ("Magnitude_y" in merged.columns)
    have_err_cols = ("Mag_Err_x" in merged.columns) and ("Mag_Err_y" in merged.columns)

    if have_mag_cols and have_err_cols:
        merged["Differential_Magnitude"] = merged["Magnitude_x"] - merged["Magnitude_y"]
        merged["Differential_Error"] = np.sqrt(
            merged["Mag_Err_x"]**2 + merged["Mag_Err_y"]**2
        )
    else:
        merged["Differential_Magnitude"] = np.nan
        merged["Differential_Error"] = np.nan

    if "Star_x" in merged.columns:
        merged.rename(columns={"Star_x": "Star"}, inplace=True)

    if "Star" not in merged.columns:
        merged["Star"] = star_id

    final = merged[["Star", "JD", "Differential_Magnitude", "Differential_Error"]]

    out_file = os.path.join(output_dir, f"Final_{star_id}_Photometry.dat")
    final.to_csv(out_file, index=False, sep='\t')

    print(f"Saved: {out_file}")

print("Done.")
