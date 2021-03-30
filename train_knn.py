import os
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import argparse
from joblib import dump, load

def paths():
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')   
    path_labels = os.path.join(INPUTS_DIR, 'labels/labels.npy')
    path_features = os.path.join(INPUTS_DIR, 'features/features.npy')
    return path_labels, path_features

def main(flags):
    path_labels, path_features = paths()
    print('Loading data ...')
    y = np.load(path_labels)
    X = np.load(path_features)
    print('Loading completed.')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Test train split completed')
    

    clf = KNeighborsClassifier(
        n_neighbors=flags.n_neighbors,
        n_jobs=flags.n_jobs
        )
    y_train=y_train.ravel()
    y_test=y_test.ravel()
    print(y.shape)
    clf.fit(X_train, y_train)
    print('Training completed')

    accuracy = clf.score(X_test, y_test)
    print('Accuracy of test:',accuracy)
    
    #kfold = KFold(n_splits=flags.n_splits)
    #print("Cross-validation scores:\n{}".format(cross_val_score(clf, X, y, cv=kfold)))

    # Get the output path from the Valohai machines environment variables
    outputs_dir = os.getenv('VH_OUTPUTS_DIR', './outputs')
    if not os.path.isdir(outputs_dir):
        os.makedirs(outputs_dir)
    save_path = os.path.join (outputs_dir, 'knn.joblib')
    dump(clf, save_path) 
    print('Model was saved')

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
    parser.add_argument(
        '--n_splits',
        type=int,
        default=3,
    )
    flags = parser.parse_args()
    return flags

if __name__ == '__main__':
    flags = parse_args()
    paths()
    main(flags)