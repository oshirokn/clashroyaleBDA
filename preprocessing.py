# 
from scripts.features import features
import os
import pandas as pd
from numpy import asarray
import numpy as np
import pandas as pd


def paths():
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')   
    folders = [
    't1/BattlesStaging_01012021_WL_tagged.csv',
    't2/BattlesStaging_01032021_WL_tagged.csv',
    't3/BattlesStaging_01042021_WL_tagged.csv',
    't4/battlesStaging_12072020_to_12262020_WL_tagged.csv',		
    't5/battlesStaging_12272020_WL_tagged.csv',		
    't6/battlesStaging_12282020_WL_tagged.csv',		
    't7/BattlesStaging_12292020_WL_tagged.csv',		
    't8/BattlesStaging_12302020_WL_tagged.csv',		
    't9/BattlesStaging_12312020_WL_tagged.csv']
    dr=[]
    for folder in folders:
        dr.append(os.path.join(INPUTS_DIR, folder))
    return dr

def main():
    print('Initializing main...')

    dr = paths() 
    chunk_size = 500000
    dfList = []
    for file in  dr:
        filename = file
        text_file_reader = pd.read_csv(filename, engine='python',encoding='utf-8-sig', chunksize = chunk_size, index_col=0)
        for df in text_file_reader:
            dfList.append(df)
    if len(dfList)>1:
        df = pd.concat(dfList,sort=False)
    else:
        df = dfList[0]
    
    print('CSV as dataframes. Calculating features and labels...')

    win_columns,loose_columns = features()

    # Features
    X1 = df[win_columns]
    X1.columns = range(X1.shape[1])
    X2 = df[loose_columns]
    X2.columns = range(X2.shape[1])
    X = pd.concat([X1,X2],axis=0)

    # Labels
    y = np.concatenate((np.ones((int(0.5*X.shape[0]),1)),
                        np.zeros((int(0.5*X.shape[0]),1))))

    print(X.values.shape,y.shape)
    print('Main done')

    return X.values,y

def save(X,y):
    print('Save features and labels as npy...')
    outputs_dir = os.getenv('VH_OUTPUTS_DIR', './outputs')
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir)
    np.save(os.path.join(outputs_dir, 'features.npy'), X)
    np.save(os.path.join(outputs_dir, 'labels.npy'), y)
    print('Save done')

def test_saved():
    
    path = os.getenv('VH_INPUTS_DIR', './inputs')
    print(os.listdir(path))



if __name__ == '__main__':    
    X,y = main()
    save(X,y)
    test_saved()



