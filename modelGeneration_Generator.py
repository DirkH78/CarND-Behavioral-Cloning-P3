import numpy as np
import csv
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D

# data conditioning
samples = []
with open('./learn/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
 
            images = []
            measurements = []
            for batch_sample in batch_samples:
                source_path_center = batch_sample[0]
                source_path_left = batch_sample[1]
                source_path_right = batch_sample[2]
                path, filename_center = os.path.split(source_path_center)
                path, filename_left = os.path.split(source_path_left)
                path, filename_right = os.path.split(source_path_right)
                current_path_center = './learn/IMG/' + filename_center
                current_path_left = './learn/IMG/' + filename_left
                current_path_right = './learn/IMG/' + filename_right
                image_center = cv2.imread(current_path_center)
                images.append(image_center)
                image_left = cv2.imread(current_path_left)
                images.append(image_left)
                image_right = cv2.imread(current_path_right)
                images.append(image_right)
                correction = 0.2 # this is a parameter to tune 
                measurement = float(batch_sample[3])
                measurements.append(measurement)
                measurements.append(measurement + correction)
                measurements.append(measurement - correction)
                    
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            yield shuffle(np.array(augmented_images), np.array(augmented_measurements))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# model definition
model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(2,2))))
model.add(Convolution2D(24, 5, 5, activation="relu"))
model.add(Convolution2D(36, 5, 5, activation="relu"))
model.add(Convolution2D(48, 5, 5, activation="relu"))
model.add(Convolution2D(64, 5, 5, activation="relu"))
model.add(Convolution2D(64, 5, 5, activation="relu"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# training
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=3, verbose = 1)
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
