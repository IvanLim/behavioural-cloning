# Generic imports
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pickle
import csv
import cv2

# Keras layers
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.callbacks import EarlyStopping

# Files and folders
FOLDER_TRAINING_DATA = './data/'
FOLDER_TRAINING_IMAGE_DATA = './data/IMG/'
FILE_DRIVING_LOG = 'driving_log.csv'
FILE_MODEL = 'model.h5'

# Settings
EPOCHS = 10
BATCH_SIZE = 32

#################
# Load raw data #
#################
# Load driving log
def load_data(file_path):

	def extract_file_name(full_path):
		return full_path.split('/')[-1]

	# Start with a fresh log
	image_files = []
	measurements = []

	print('Loading training data...')
	with open(file_path) as csv_file:
		reader = csv.reader(csv_file)
		headers = next(reader)
		counter = 0

		# The angles seem to be recorded in the format: DEGREES / 25.0
		# If we want to correct by 3.1 degrees, we need to correct by 3.1 / 25 = 0.124
		correction = 0.124
		for line in reader:
			# Get the images for all 3 angles
			img_center = extract_file_name(line[0])
			img_left = extract_file_name(line[1])
			img_right = extract_file_name(line[2])

			# Estimate the steering angles for the left and right cameras
			angle_center = float(line[3])
			angle_left = angle_center + correction
			angle_right = angle_center - correction

			# Add them to the training set
			image_files.extend([img_center, img_left, img_right])
			measurements.extend([angle_center, angle_left, angle_right])
	print('\tDONE -> {} entries read.'.format(len(image_files)))
	print()

	return shuffle(np.array(image_files), np.array(measurements))


####################################
# Generate data for training batch #
####################################
# Loads one batch_size worth of raw training data each time, 
# until we've gone through all the training examples
def batch_generator(image_files, measurements, batch_size):
	num_measurements = len(measurements)

	while True:
		batch_image_files = []
		batch_measurements = []

		for offset in range(0, num_measurements, batch_size):
			end = offset + batch_size			
			batch_image_files = image_files[offset:end]			
			batch_measurements = measurements[offset:end]

			# Load the images for this batch
			batch_images = []
			for images in batch_image_files:
				center = cv2.imread(FOLDER_TRAINING_IMAGE_DATA + images)
				center_resized = cv2.resize(center, (160, 80))
				batch_images.append(center_resized)

			batch_measurements_flipped = [-1 * x for x in batch_measurements]
			batch_images_flipped = [np.fliplr(x) for x in batch_images]
			
			yield (np.concatenate((batch_images, batch_images_flipped)), 
			 	   np.concatenate((batch_measurements, batch_measurements_flipped)))


####################
# Model Definition #
####################
def build_model():
	model = Sequential()
	model.add(Cropping2D(cropping=((25, 12), (0, 0)), input_shape=(80, 160, 3)))
	model.add(Lambda(lambda x: x / 127.5 - 1.))	
	model.add(Convolution2D(3, 1, 1))
	model.add(Convolution2D(18, 5, 5))
	model.add(Convolution2D(24, 3, 3))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(550))
	model.add(Activation('relu'))
	model.add(Dense(100))
	model.add(Activation('relu'))
	model.add(Dense(80))
	model.add(Activation('relu'))
	model.add(Dense(1))
	return model

############
# Training #
############
def train_model(model, train_generator, num_train_samples, validation_generator, num_validation_samples, num_epochs):
	print("Training...")

	# We want to stop training the moment there is no improvement
	early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')

	model.compile(loss='mse', optimizer='adam')
	model.fit_generator(train_generator, 
						samples_per_epoch=num_train_samples,
						validation_data=validation_generator,
						nb_val_samples=num_validation_samples,
						callbacks=[early_stop],
						nb_epoch=num_epochs)
	print("\t-> Done")
	print()
	print("Saving model...")
	model.save(FILE_MODEL)
	print("\t-> Done")
	print()


##############
# Evaluation #
##############
def evaluate_model(model, test_generator, num_test_samples):
	print("Evaluating model against test set...")
	metric_value = model.evaluate_generator(generator=test_generator, val_samples=num_test_samples)
	for i in range(len(model.metrics_names)):
		metric_name = model.metrics_names[i]		
		print('\t{}: {}'.format(metric_name, metric_value))
	print("\t-> Done")
	print()


##############
# Main logic #
##############
# Load the raw data and split it into training and test sets
image_files, measurements = load_data(FOLDER_TRAINING_DATA + FILE_DRIVING_LOG)
train_image_files, remaining_image_files, train_measurements, remaining_measurements = train_test_split(image_files, measurements, test_size=0.1)
test_image_files, validation_image_files, test_measurements, validation_measurements = train_test_split(remaining_image_files, remaining_measurements, test_size=0.5)

# Build, train, and evaluate the model
model = build_model()

# For each car position captured we have 3 images (center, left and right)
# Also, the data is augmented by flipping, so we actually have 3 * 2 the number of samples
num_train_samples = len(train_measurements) * 3 * 2
num_validation_samples = len(validation_measurements) * 3 * 2
num_test_samples = len(test_measurements) * 3 * 2

# Set up our training, validation, and test generators
train_generator = batch_generator(train_image_files, train_measurements, BATCH_SIZE)
validation_generator = batch_generator(validation_image_files, validation_measurements, BATCH_SIZE)
test_generator = batch_generator(test_image_files, test_measurements, BATCH_SIZE)

train_model(model, train_generator, num_train_samples, validation_generator, num_validation_samples, EPOCHS)
evaluate_model(model, test_generator, num_test_samples)