import numpy as np 
import pandas as pd
import os
import csv

def paths():
    root = "."
    dr=[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            #         print(os.path.join(path, name))
            _= os.path.join(path, name)
            if _.endswith('.csv'):
                dr.append(_)
    return dr
dr = paths()


dfList = []
for filename in dr:
    text_file_reader = pd.read_csv(filename, engine='python',encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL, chunksize = 500000, index_col=0)
    counter = 0
    for df in text_file_reader:
        dfList.append(df)
        counter= counter +1
        print("Max rows read: " + str(chunk_size * counter) )
df = pd.concat(dfList,sort=False)
print(df)

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
