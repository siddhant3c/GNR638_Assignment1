import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
from PIL import Image

# Read the TIF image
image = imread('../reference/UCMerced_LandUse/Images/airplane/airplane01.tif')
# image = Image.open('reference/Scene-Recognition-with-Bag-of-Words-master/data/test/Bedroom/image_0003.jpg')

# convert image to numpy array
image = np.array(image)

# Print image shape (optional)
print("Image shape:", image.shape)
# print(image)

# Change the order of the channels from RGB to BGR
image = image[:, :, ::-1]

# separate the image along the 3 channels
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

# Display the image and the 3 channels
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(red_channel, cmap='gray')
plt.title('Red Channel')

plt.subplot(2, 2, 3)
plt.imshow(green_channel, cmap='gray')
plt.title('Green Channel')

plt.subplot(2, 2, 4)
plt.imshow(blue_channel, cmap='gray')
plt.title('Blue Channel')

plt.show()