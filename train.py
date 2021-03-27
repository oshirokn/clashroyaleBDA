import os
import numpy as np

def paths():
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')   
    path_labels = os.path.join(INPUTS_DIR, 'labels/labels.npy')
    return path_labels

def main():
    path_labels = paths()
    print('Loading data ...')
    y = np.load(path_labels)
    print('Loading completed.')
    print(y[:10,:])

if __name__ == '__main__': 
    paths()
    main()