# Import for project required libraries/functions
import cv2
import csv
import sklearn

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from math import ceil

# Tensorflow and keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D





# Relative path to the image-directory which contains all images recorded within the simulator provided by Udacity
imageBasePath = './next_data/IMG/'

# Value to add/substract from the steering-value for the left-/right-ward shifted images
correction_value = 0.2
# Array to store the readin and generated images
data = []

# Iterate though all given recorded images and add the values into the data-array
with open('./next_data/driving_log_next.csv', 'r') as csvfile:
    csvRows = csv.reader(csvfile, delimiter=',')
    for row in csvRows:
        # 6 Images are added, the centered, leftward-offset and rightward-offset image. All of them are added twice,
        # because the second version of then is added and will be (flipped) mirrored-horizontal later
        # Therefore the third value of the index is added:
        # (0|1) 1 = image as is | 0 = image to be flipped and steering to be negated
        data.append([row[0].split('\\')[-1], float(row[3]), 1]) # Centered camera image with normal steering value
        data.append([row[1].split('\\')[-1], float(row[3]) + correction_value, 1]) # Leftward offset image with higher steering-value to turn-right
        data.append([row[2].split('\\')[-1], float(row[3]) - correction_value, 1]) # Rightward offset image with lower steering-value to turn-left
        # Same as above just with the additional third value as 0 instead of 1 as described above
        data.append([row[0].split('\\')[-1], float(row[3]), 0])
        data.append([row[1].split('\\')[-1], float(row[3]) + correction_value, 0]) 
        data.append([row[2].split('\\')[-1], float(row[3]) - correction_value, 0]) 
        
# Data generator to avoid to load the whole available data (+ generated data) into the memory.
# This may be slower but avoid a out-of-memory exception
# The code is inspired by the provided code in the udacity lecture "Project Behavioral Cloning" -> "18. Generator"
def generator(samples, batch_size=32):
    # Store amount of data
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # Shuffle data for each epoch
        sklearn.utils.shuffle(samples)
        # Iterate though data. 
        for offset in range(0, num_samples, batch_size):
            # The next batch from full list of data
            batch_samples = data[offset:offset + batch_size]

            # Images and steering-values will be saved independently
            images = []
            angles = []
            # Add required amount of images and steering-values to specific lists
            for batch_sample in batch_samples:
                if batch_sample[2] == 1: # Take original image and only convert from BGR to RGB
                    image = cv2.imread(imageBasePath + batch_sample[0])
                    cvtImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(cvtImage)
                    angles.append(batch_sample[1])
                else: # Take the copy of the images (each image was added twice) and flip it horizontal (data-generation)
                    image = cv2.imread(imageBasePath + batch_sample[0])
                    cvtImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Flip horizontal (mirroring)
                    images.append(np.fliplr(cvtImage))
                    # Negate steering values
                    angles.append(batch_sample[1] * -1)
                    
            # List to numpy array
            X_train = np.array(images)
            y_train = np.array(angles)
            
            # Return data within yield for fit_generator to keep iterating through images if .next is called
            yield X_train, y_train

# Batch size to train with
batch_size=128

# Split data 80:20  80 = Training, 20 = Validation
train_samples, validation_samples = train_test_split(data, test_size=0.2)


# Create generators for train and validation data using the generator-function above
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# Create sequential model
model = Sequential()

# Preprocess image by copping top 50, bottom 20 pixels
# and change each pixel-value-range vom 0-255 to -0.5 - +0.5
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# The network is inspired by the following two architectures:
# LeNet: https://miro.medium.com/max/4308/1*1TI1aGBZ4dybR6__DI9dzA.png
# NVIDIA - Self driving car network: https://miro.medium.com/max/3236/1*HwZvJLpALucQkBuBCFDxKw.png
# First conv layer. 
model.add(Convolution2D(6, 5, 5, subsample=(2,2), activation='relu'))
# Dropout 30% to avoid overfitting
model.add(Dropout(0.3))
# Second conv layer. 
model.add(Convolution2D(32, 5, 5, subsample=(2,2), activation='relu'))
# Dropout 30% to avoid overfitting
model.add(Dropout(0.3))
# Third conv layer. 
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
# Dropout 30% to avoid overfitting
model.add(Dropout(0.3))
# Fourth. 
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
# Dropout 30% to avoid overfitting
model.add(Dropout(0.3))
# Flatten the output of the fourth conv layer
model.add(Flatten())

# Add 4 additional fully connected layers
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(10))
model.add(Dense(1))


# Use mean square error as loss-function and adam-optimizer for the model-compilation
model.compile(loss='mse', optimizer='adam')

# Use fit_generator-function to use the generators above to train/validate the model
# This function is provided by keras and automaitcally iterates through the training-/validation-data using the generators
# The generators are required to avoid the network to load all of the training data (>10k images into memory)
# By adjusting the batchsize you can configure, how much data to load into memory while training
model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples) / batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples) / batch_size), 
epochs=3, verbose=1)

# Save the trained model to use within the drive.py script in to simulator
model.save('./model3.h5')