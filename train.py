import os

def main():
    path = os.getenv('VH_INPUTS_DIR', './inputs')
    print(os.listdir(path))

if __name__ == '__main__': 
    main()