# convert all files in the "data" directory from .tif to .jpg

import os
from PIL import Image
from tqdm import tqdm

data_path = "../data"

def convert_tif_to_jpg(data_path):
    # visualize progress using tqdm
    for root, dirs, files in os.walk(data_path):
        for file in tqdm(files):
            if file.endswith(".tif"):
                img = Image.open(os.path.join(root, file))
                img.save(os.path.join(root, file.replace(".tif", ".jpg")))

def remove_tif_files(data_path):
    # remove all .tif files in the "data" directory
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".tif"):
                os.remove(os.path.join(root, file))

if __name__ == "__main__":
    convert_tif_to_jpg(data_path)
    remove_tif_files(data_path)