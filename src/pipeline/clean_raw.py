import pandas as pd 

import glob
import os
path = '../../data/raw/box/'
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

#clean up data, remove spaces
#concatenate values for column 8 (text field)
def get_clean_df(filename):
    filepath = '../../data/raw/box/'+filename
    to_write_filepath = '../../data/interim2/'+filename
    df = pd.read_csv(filepath, header=None, sep='\n')
    df = df[0].str.split(',', expand=True)
    temp = df.copy() 
    temp[temp.columns] = temp.apply(lambda x: x.str.strip())
    temp.fillna('', inplace=True)
    temp[8]= temp[8].str.cat(temp.iloc[:,9:], sep =", ") 
    temp[temp.columns] = temp.apply(lambda x: x.str.rstrip(", ,"))
    temp = temp.loc[:, :8]
    temp.to_csv(to_write_filepath, index=False) #,float_format='%f')
    return temp

#print(get_clean_df('000.csv'))

for files in csv_files:
    get_clean_df(files)
