# split the UCMercey dataset into training and testing data
# with this snippet from Assignment1/utils/split_data.py:

import os
from tqdm import tqdm

# Target directory
TARGER_DIR = '../data'

# Load data
DATA_DIR = '../reference/UCMerced_LandUse/Images'

# each folder in the data directory is a class
# each class has 100 images

if __name__ == '__main__':
    # create a list of all the classes
    classes = os.listdir(DATA_DIR)

    # sort classes in alphabetical order
    classes.sort()

    # save the data to the data directory, under train & test, and visualize progress using tqdm
    os.makedirs(f'{TARGER_DIR}/train', exist_ok=True)
    os.makedirs(f'{TARGER_DIR}/test', exist_ok=True)

    # create the class directories in the train and test directories
    for c in classes:
        os.makedirs(f'{TARGER_DIR}/train/{c}', exist_ok=True)
        os.makedirs(f'{TARGER_DIR}/test/{c}', exist_ok=True)

    # save 80% of the images to the train directory and 20% to the test directory
    for c in tqdm(classes):
        files = os.listdir(f'{DATA_DIR}/{c}')
        n = len(files)
        n_train = int(0.8 * n)
        n_test = n - n_train
        train_files = files[:n_train]
        test_files = files[n_train:]
        for f in train_files:
            os.system(f'cp {DATA_DIR}/{c}/{f} {TARGER_DIR}/train/{c}/{f}')
        for f in test_files:
            os.system(f'cp {DATA_DIR}/{c}/{f} {TARGER_DIR}/test/{c}/{f}')

    print('Data split complete')
