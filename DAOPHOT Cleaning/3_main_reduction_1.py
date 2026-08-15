import os
import pandas as pd

root_directory = "star_files_1"
output_directory = "star_files_2"
MIN_ROWS = 1000

os.makedirs(output_directory, exist_ok=True)

kept_files = []

for filename in os.listdir(root_directory):
    input_path = os.path.join(root_directory, filename)

    if not os.path.isfile(input_path):
        continue

    try:
        df = pd.read_csv(input_path, sep='\s+', engine="python") #type:ignore

        if len(df) > MIN_ROWS:

            if "NL" in df.columns:
                df = df.drop(columns=["NL"])

            kept_files.append(df)

    except Exception as e:
        print(f"Skipping {filename}: {e}")

for i, df in enumerate(kept_files, start=1):
    output_path = os.path.join(output_directory, f"{i}.dat")
    df.insert(0,'Star',i)
    df.to_csv(output_path, index=False)
