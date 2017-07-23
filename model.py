import csv
import cv2
import sklearn
import numpy as np
import os.path
from sklearn.model_selection import train_test_split

## first, read in the training data's csv file
lines = []
num_lines = 0

with open('../windows_sim/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# read now all images (central cam, right and left cam, and do augmentation)
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset + batch_size]
			images = []
			measurements = []
			for batch_sample in batch_samples:
				current_path = batch_sample[0] # image of center camera
				image = cv2.imread(current_path)
				images.append(image)
				mesurement = float(batch_sample[3]) # steering angle for center camera
				measurements.append(mesurement)
				image = cv2.flip(image, 1) #  image of center camera, horizontal flip
				images.append(image)
				measurements.append(-1.0 * mesurement) # steering angle for flipped image
				images.append(cv2.imread(batch_sample[1])) # image of left camera
				measurements.append(mesurement + 0.2) # correction = +0.2
				images.append(cv2.imread(batch_sample[2])) # image if right camera
				measurements.append(mesurement - 0.2) # correction = -0.2
		# convert arrays to numpy arrays
			X_train = np.array(images)
			y_train = np.array(measurements)
			#print(X_train.shape)
			#print(y_train.shape)
			yield sklearn.utils.shuffle(X_train, y_train)

		
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

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

#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4, verbose=1)

from keras.models import Model
import matplotlib.pyplot as plt
history_object = model.fit_generator(train_generator, samples_per_epoch =len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch=3, verbose=1)

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