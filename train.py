import os
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import argparse

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
    

    clf = KNeighborsClassifier(
        n_neighbors=flags.n_neighbors,
        n_jobs=flags.n_jobs
        )
    clf.fit(X, y.ravel())
    print('Training completed')
    
    kfold = KFold(n_splits=5)
    print("Cross-validation scores:\n{}".format(cross_val_score(clf, X, y, cv=kfold)))

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