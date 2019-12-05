# This script is only used to create figures which are used in the documentation
# The script is hardcoded and not a good example how to develop :-)

# Dependencies
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg') # Do not display (no display in console at udacity)
import matplotlib.pyplot as plt


# Example data from next './next_data/driving_log_next.csv'
data = [
    ['./next_data/IMG/center_2019_11_30_02_59_02_499.jpg', './next_data/IMG/left_2019_11_30_02_59_02_499.jpg', './next_data/IMG/right_2019_11_30_02_59_02_499.jpg', 0.0],
    ['./next_data/IMG/center_2019_11_30_03_01_49_365.jpg', './next_data/IMG/left_2019_11_30_03_01_49_365.jpg', './next_data/IMG/right_2019_11_30_03_01_49_365.jpg', 0.5022397]
]

# Get images to display (also convert from BGR to RGB)
img1_center = cv2.cvtColor(cv2.imread(data[0][0]), cv2.COLOR_BGR2RGB)
img1_left = cv2.cvtColor(cv2.imread(data[0][1]), cv2.COLOR_BGR2RGB)
img1_right = cv2.cvtColor(cv2.imread(data[0][2]), cv2.COLOR_BGR2RGB)

img2_center = cv2.cvtColor(cv2.imread(data[1][0]), cv2.COLOR_BGR2RGB)
img2_left = cv2.cvtColor(cv2.imread(data[1][1]), cv2.COLOR_BGR2RGB)
img2_right = cv2.cvtColor(cv2.imread(data[1][2]), cv2.COLOR_BGR2RGB)


# Horizontally flip images
img1_center_flipped = np.fliplr(img1_center)
img1_left_flipped = np.fliplr(img1_left)
img1_right_flipped = np.fliplr(img1_right)

img2_center_flipped = np.fliplr(img2_center)
img2_left_flipped = np.fliplr(img2_left)
img2_right_flipped = np.fliplr(img2_right)


# Visualize images in figure and save it afterwards
fig, subplots = plt.subplots(2,3, figsize=(20, 8))
subplots[0][0].imshow(img1_center)
subplots[0][0].set_title("Centered image #1")
subplots[0][1].imshow(img1_left)
subplots[0][1].set_title("Left image #1 (+2.0 steering angle)")
subplots[0][2].imshow(img1_right)
subplots[0][2].set_title("Right image #1 (-2.0 steering angle)")
subplots[1][0].imshow(img2_center)
subplots[1][0].set_title("Centered image #2")
subplots[1][1].imshow(img2_left)
subplots[1][1].set_title("Left image #2 (+2.0 steering angle)")
subplots[1][2].imshow(img2_right)
subplots[1][2].set_title("Right image #2 (-2.0 steering angle)")

fig.savefig("./example_figures/example-images.png")


# Visualize flipped images in figure and save it afterwards
fig, subplots = plt.subplots(2,3, figsize=(20, 8))
subplots[0][0].imshow(img1_center_flipped)
subplots[0][0].set_title("Centered image #1 flipped")
subplots[0][1].imshow(img1_left_flipped)
subplots[0][1].set_title("Left image #1 (flipped)")
subplots[0][2].imshow(img1_right_flipped)
subplots[0][2].set_title("Right image #1 (flipped)")
subplots[1][0].imshow(img2_center_flipped)
subplots[1][0].set_title("Centered image #2")
subplots[1][1].imshow(img2_left_flipped)
subplots[1][1].set_title("Left image #2 (flipped)")
subplots[1][2].imshow(img2_right_flipped)
subplots[1][2].set_title("Right image #2 (flipped)")

fig.savefig("./example_figures/example-images-flipped.png")