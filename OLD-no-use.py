import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np 
import pandas as pd
import os
import csv
import argparse
import json
from joblib import dump, load
from sklearn.model_selection import KFold
#from parse_layer_spec import add_layers
#from utils import use_valohai_inputs

def paths():
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')

    _ = os.listdir(INPUTS_DIR)
    
    file_ = [
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
            #print("Max rows read: " + str(chunk_size * counter) )
    df = pd.concat(dfList,sort=False)
    
    
    print(df.shape, df.memory_usage(index=True).sum())
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
    
    model = KNeighborsClassifier(
        n_neighbors=flags.n_neighbors,
        n_jobs=flags.n_jobs
        )
    model.fit(X_train, y_train.values.ravel())
    print('Training completed')
    
    accuracy = model.score(X_test, y_test)
    print('Accuracy of test:',accuracy)
    kfold = KFold(n_splits=5)
    print("Cross-validation scores:\n{}".format(cross_val_score(model, X, y, cv=kfold)))
    
    # Get the output path from the Valohai machines environment variables
    outputs_dir = os.getenv('VH_OUTPUTS_DIR', './outputs')
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir)
    save_path = os.path.join (outputs_dir, 'test-kn.joblib')
    dump(model, save_path) 
    print('Model was saved')

    {'criterion': 'entropy',
 'max_depth': 25,
 'min_samples_leaf': 1,
 'n_estimators': 50,
 'n_jobs': 4}
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_neighbors',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=8,
    )
    flags = parser.parse_args()
    return flags

if __name__ == '__main__':    
    flags = parse_args()
    paths()
    main(flags)
