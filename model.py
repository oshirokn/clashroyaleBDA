import numpy as np 
import pandas as pd
import os
import csv

for dirname, _, filenames in os.getcwd():
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

csv_filename = ".\kaggle\input\clash-royale-season-18-dec-0320-dataset\BattlesStaging_01012021_WL_tagged.csv"
chunk_size = 500000

cardmasterlist_csv_filename = "/CardMasterListSeason18_12082020.csv"
df_cardmasterlist = pd.read_csv(cardmasterlist_csv_filename, engine='python',encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL, index_col = 0)
df_cardmasterlist.reset_index(inplace=True)
cardmasterlist_dict = dict(zip(df_cardmasterlist["team.card1.id"],df_cardmasterlist["team.card1.name"]))

text_file_reader = pd.read_csv(csv_filename, engine='python',encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL, nrows = n_rows, chunksize = chunk_size, index_col=0)
dfList = []
counter = 0

for df in text_file_reader:
    dfList.append(df)
    counter= counter +1
    print("Max rows read: " + str(chunk_size * counter) )
df = pd.concat(dfList,sort=False)

columns = ['winner.card1.id', 'winner.card2.id','winner.card3.id', 'winner.card4.id','winner.card5.id', 'winner.card6.id','winner.card7.id', 'winner.card8.id']
X = df[columns]
X.columns = ['card1',"card2","card3","card4","card5","card6","card7","card8"]
columns = ['loser.card1.id', 'loser.card2.id','loser.card3.id', 'loser.card4.id','loser.card5.id', 'loser.card6.id', 'loser.card7.id', 'loser.card8.id'] 
X2 = df[columns]
X2.columns = ['card1',"card2","card3","card4","card5","card6","card7","card8"]
X= pd.concat([X, X2], ignore_index=True, sort=True)
print(X)

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
print(X)

columns = ['winner.trophyChange']
L = df[columns]
y = pd.DataFrame().reindex_like(L)
y = y.fillna(1)
y2 = pd.DataFrame().reindex_like(L)
y2 = y2.fillna(0)
y = pd.concat([y, y2], ignore_index=True, sort=True)
y.columns = ['result']
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_test.shape)
print(X_train.shape)
print(y_train.shape)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train.values.ravel())
accuracy = model.score(X_test, y_test)
print(accuracy)
