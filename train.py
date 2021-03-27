import os

def test_saved():
    
    path = os.getenv('VH_INPUTS_DIR', './inputs')
    print(os.listdir(path))

if __name__ == '__main__': 
    test_saved()