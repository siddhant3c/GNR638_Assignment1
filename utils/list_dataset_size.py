# List down the number of images in each subfolder (Class) of the UCMerced_LandUse dataset.

import os

DATASET_PATH = '../reference/UCMerced_LandUse/Images'
CATEGORIES = os.listdir(DATASET_PATH)

for category in CATEGORIES:
    category_path = os.path.join(DATASET_PATH, category)
    print(f"{category}: {len(os.listdir(category_path))} images")

# Output:
# agricultural: 100 images
# airplane: 100 images
# baseballdiamond: 100 images
# beach: 100 images
# buildings: 100 images
# ...

# The output shows the number of images in each subfolder (Class) of the UCMerced_LandUse dataset.
