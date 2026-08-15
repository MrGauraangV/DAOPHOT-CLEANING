import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

root_directory = 'star_files_2'
file_paths = [os.path.join(root_directory, f) for f in glob.glob('*.dat', root_dir=root_directory)]
master_df = pd.concat([pd.read_csv(f, delimiter=',') for f in file_paths], ignore_index=True)#type:ignore
master_df.columns = ['Star', 'JD','NX','NY','Magnitude','Mag_Err']
corrected = master_df[master_df['Mag_Err'] < 0.25]

import glob
import os
import matplotlib.pyplot as plt

root_directory = 'star_files_2'
file_paths = [os.path.join(root_directory, f) for f in glob.glob('*.dat', root_dir=root_directory)]

master_df = pd.concat([pd.read_csv(f, delimiter=',') for f in file_paths], ignore_index=True) #type:ignore
master_df.columns = ['Star', 'JD','NX','NY','Magnitude','Mag_Err']

corrected = master_df[master_df['Mag_Err'] < 0.25]

corrected['Mag_bin'] = corrected['Magnitude'].round(2)

avg_df = corrected.groupby('Mag_bin')['Mag_Err'].mean().reset_index()

plt.scatter(avg_df['Mag_bin'], avg_df['Mag_Err'], s=10, alpha=0.15)
plt.xlim(12,18)
plt.xlabel('Magnitude')
plt.ylabel('Average Mag Error')
plt.show()
