import csv
import cv2
import numpy as np

lines = []
with open('../Driving_Data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
correction = 0.2
for line in lines:
    # Center image path
    source_path = line[0]
    filename_center = source_path.split('/')[-1]
    # Left image path
    source_path = line[1]
    filename_left = source_path.split('/')[-1]
    # Right image path
    source_path = line[2]
    filename_right = source_path.split('/')[-1]
    # Load center images
    current_path = '../Driving_Data2/IMG/' + filename_center
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    images.append(cv2.flip(image,1))    # Add flipped image
    measurements.append(measurement*-1.0)    # Add flipped steering angle
    # Load left images
    current_path = '../Driving_Data2/IMG/' + filename_right
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement - correction)
    images.append(cv2.flip(image,1))    # Add flipped image
    measurements.append((measurement - correction)*-1.0)    # Add flipped steering angle
    # Load right Images
    current_path = '../Driving_Data2/IMG/' + filename_left
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    images.append(cv2.flip(image,1))    # Add flipped image
    measurements.append((measurement + correction)*-1.0)    # Add flipped steering angle



X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Convolutional Model
model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5)) #, input_shape=(160,320,3)))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Train the network
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch=5)

#Save the model
model.save('model.h5')
exit()
