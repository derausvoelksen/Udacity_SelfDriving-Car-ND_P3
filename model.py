import csv
import cv2
import numpy as np
import os.path


## first, read in the training data's csv file
lines = []
num_lines = 0

with open('../windows_sim/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []


# read now all images (central cam, right and left cam, and do augmentation)
for line in lines:
    current_path = line[0] # image of center camera
    image = cv2.imread(current_path)
    images.append(image)

    image = cv2.flip(image, 1) #  image of center camera, horizontal flip
    images.append(image)

    mesurement = float(line[3]) # steering angle for center camera
    measurements.append(mesurement)
    measurements.append(-1.0 * mesurement) # steering angle for flipped image

    images.append(cv2.imread(line[1])) # image of left camera
    measurements.append(mesurement + 0.2) # correction = +0.2

    images.append(cv2.imread(line[2])) # image if right camera
    measurements.append(mesurement - 0.2) # correction = -0.2
    num_lines+=1
    if num_lines%500 == 0:
        print('Number of lines already processed:', num_lines)

# convert arrays to numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)


def rgb_to_greyscale(image):
	from keras.backend import tf as ktf
	return ktf.image.rgb_to_grayscale(image)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout

## nvidia model
model = Sequential()
#model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Lambda(rgb_to_greyscale, input_shape = (160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Cropping2D(((70,25),(0,0))))
model.add(Convolution2D(24, 5,5, subsample= (2,2), activation='relu'))
model.add(Convolution2D(36, 5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3,3, subsample=(1,1), activation='relu'))
model.add(Convolution2D(64, 3,3, subsample=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#print(X_train.shape)

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)
model.save('model.h5')