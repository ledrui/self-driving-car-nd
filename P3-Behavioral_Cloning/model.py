import csv
import cv2
import numpy as np
import sklearn


lines = []
with open('./data/driving_log.csv') as csvfile:
    next(csvfile, None)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    for i in range(3):
        # Load images from center, left and right cameras
        source_path = line[i]
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = "./data/IMG/" + filename
        image = cv2.imread(local_path)
        images.append(image)

    # Introduce steering correction
    correction = 0.2
    measurement = float(line[3])
    # Steering adjustment for center images
    measurements.append(measurement)
    # Add correction for steering for left images
    measurements.append(measurement+correction)
    # Minus correction for steering for right images
    measurements.append(measurement-correction)

augmented_images = []
augmented_measurements = []

# Augmented data set by adding 'flipped' images
# so model can learn from reversed images,
#as well as random brightness
#(with thanks to Vivek Yadav at http://bit.ly/2kOk6MU for the latter)
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    brightened_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    brightened_image[:,:,2] = brightened_image[:,:,2]*random_bright
    brightened_image = cv2.cvtColor(brightened_image,cv2.COLOR_HSV2RGB)
    flipped_image = cv2.flip(brightened_image, 1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

# Pull the image and steering measurements
# into NumPy arrays we can use in the model
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, convolutional, core, pooling

# NVIDIA end to end Model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
model.add(core.Dense(500, activation='relu'))
model.add(core.Dense(100, activation='relu'))
model.add(core.Dense(20, activation='relu'))
model.add(core.Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)
model.save('model.h5')
