import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

data = glob.glob('*.ref',
                root_dir='rfilter_refs')
output_dir = 'clean_files'

JD_data = pd.read_csv('rfilter_refs/JD_all.dat', delimiter='\s+') #type:ignore
pattern = r'(\d{8}).*?(\d{4})\.ref'
JD_data[['Date', 'Number']] = JD_data['Name'].str.extract(pattern)


file_dates = [os.path.basename(f)[:8] for f in data]
date_frame = [os.path.basename(f)[-8:-4] for f in data]


for n in range(len(data)):
    df = pd.read_csv('rfilter_refs/'+data[n],delimiter='\s+') #type:ignore
    clean = df.iloc[1:, [0,1,2,3,4]]
    clean.insert(0,'DATE',file_dates[n])
    clean.insert(1,'FRAME',date_frame[n])
    clean.columns = ['DATE', 'FRAME', 'NL','NX','NY', 'Magnitude', 'Mag_Err']

    new_file_name = file_dates[n] + '_' + date_frame[n] + '.csv'
    full_path = os.path.join(output_dir, new_file_name)

    testdata = clean.sort_values(by='NL')

    testdata['DATE'] = testdata['DATE'].astype(str)
    testdata['FRAME'] = testdata['FRAME'].astype(str).str.zfill(4)

    merged = pd.merge(testdata, JD_data[['JD','Date','Number']], how='left', left_on=['DATE', 'FRAME'], right_on=['Date', 'Number'])
    final = merged.drop(columns=['Date', 'Number']).loc[:,['JD', 'NL', 'NX', 'NY', 'Magnitude', 'Mag_Err']]
    final.to_csv(full_path,index=False)

