
import numpy as np 
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
import os
import csv
import argparse
import json
#from parse_layer_spec import add_layers
#from utils import use_valohai_inputs

def paths():
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')

    _ = os.listdir(INPUTS_DIR)
    
    file_ = [
    't1/BattlesStaging_01012021_WL_tagged.csv',
    't2/BattlesStaging_01032021_WL_tagged.csv']

    dr=[]
    for folder in file_:
        dr.append(os.path.join(INPUTS_DIR, folder))
    return dr



        
def main(flags):
    
    dr = paths() 
    #print(os.listdir(os.getenv('VH_INPUTS_DIR', './inputs')))
    chunk_size = 500000
    dfList = []
    for file in  dr:
        filename = file
        text_file_reader = pd.read_csv(filename, engine='python',encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL, chunksize = chunk_size, index_col=0)
        counter = 0
        for df in text_file_reader:
            dfList.append(df)
            counter= counter +1
            print("Max rows read: " + str(chunk_size * counter) )
    df = pd.concat(dfList,sort=False)
    
    
    print('Data is loaded and stored into a dataframe')

    columns = ['winner.card1.id', 'winner.card2.id','winner.card3.id', 'winner.card4.id','winner.card5.id', 'winner.card6.id','winner.card7.id', 'winner.card8.id']
    X = df[columns]
    X.columns = ['card1',"card2","card3","card4","card5","card6","card7","card8"]
    columns = ['loser.card1.id', 'loser.card2.id','loser.card3.id', 'loser.card4.id','loser.card5.id', 'loser.card6.id', 'loser.card7.id', 'loser.card8.id'] 
    X2 = df[columns]
    X2.columns = ['card1',"card2","card3","card4","card5","card6","card7","card8"]
    X= pd.concat([X, X2], ignore_index=True, sort=True)

    columns = ["winner.totalcard.level","winner.troop.count",'winner.structure.count', 'winner.spell.count', 'winner.common.count',
     'winner.rare.count', 'winner.epic.count', 'winner.legendary.count']
    X2 =df[columns]
    X2.columns = ["totalcard.level","troop.count",'structure.count', 'spell.count', 'common.count',
     'rare.count', 'epic.count', 'legendary.count']
    columns = ["loser.totalcard.level","loser.troop.count",'loser.structure.count', 'loser.spell.count', 'loser.common.count',
     'loser.rare.count', 'loser.epic.count', 'loser.legendary.count']
    X3 =df[columns]
    X3.columns= ["totalcard.level","troop.count",'structure.count', 'spell.count', 'common.count',
     'rare.count', 'epic.count', 'legendary.count']
    X2= pd.concat([X2, X3], ignore_index=True, sort=True)
    X= X.join(X2)

    columns = ['winner.trophyChange']
    L = df[columns]
    y = pd.DataFrame().reindex_like(L)
    y = y.fillna(1)
    y2 = pd.DataFrame().reindex_like(L)
    y2 = y2.fillna(0)
    y = pd.concat([y, y2], ignore_index=True, sort=True)
    y.columns = ['result']

    print('Features and labels are ready now')
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Test train split completed')
    
    model = SVC(C=flags.C, gamma=flags.gamma, n_jobs=8)
    model.fit(X_train, y_train.values.ravel())
    print('Training completed')
    
    accuracy = model.score(X_test, y_test)
    print('Accuracy of test:',accuracy)
    
    # Get the output path from the Valohai machines environment variables
    output_path = os.getenv('VH_OUTPUTS_DIR')
    model.save(os.path.join(output_path, 'model.h5'))
    print('Model was saved')
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--C',
        type=float,
        default=1,
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=10,
    )
    flags = parser.parse_args()
    return flags

if __name__ == '__main__':    
    flags = parse_args()
    paths()
    main(flags)
