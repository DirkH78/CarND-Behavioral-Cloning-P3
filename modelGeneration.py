import numpy as np
import matplotlib.image as mpimg
import csv
import cv2
import os
import matplotlib.image as mpimg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D

# data conditioning
lines = []
with open('./learn/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    source_path_center = line[0]
    source_path_left = line[1]
    source_path_right = line[2]
    path, filename_center = os.path.split(source_path_center)
    path, filename_left = os.path.split(source_path_left)
    path, filename_right = os.path.split(source_path_right)
    current_path_center = './learn/IMG/' + filename_center
    current_path_left = './learn/IMG/' + filename_left
    current_path_right = './learn/IMG/' + filename_right
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune

    image_center = mpimg.imread(current_path_center)
    images.append(image_center)
    image_left = mpimg.imread(current_path_left)
    images.append(image_left)
    image_right = mpimg.imread(current_path_right)
    images.append(image_right)
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement + correction)
    measurements.append(measurement - correction)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# model definition
model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5 ,subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5 ,subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5 ,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
#model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# training
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, verbose = 1, nb_epoch = 4)
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
