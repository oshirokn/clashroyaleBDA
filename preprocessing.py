# 
import os
from scripts.features import features

def paths():
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')

    _ = os.listdir(INPUTS_DIR)
    
    file_ = [
    't1/BattlesStaging_01012021_WL_tagged.csv']

    dr=[]
    for folder in file_:
        dr.append(os.path.join(INPUTS_DIR, folder))
    return dr

def main(dr):

     dr = paths() 
    chunk_size = 500000
    dfList = []
    for file in  dr:
        filename = file
        text_file_reader = pd.read_csv(filename, engine='python',encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL, chunksize = chunk_size, index_col=0)
        for df in text_file_reader:
            dfList.append(df)
    if len(dfList)>1:
        df = pd.concat(dfList,sort=False)
    else:
        df = dfList[0]

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
    return X,y

def save():


if __name__ == '__main__':    
    dr = paths()
    main(dr)


