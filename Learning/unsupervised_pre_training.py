# unsupervised greedy layer-wise pretraining for blobs classification problem
#from: https://machinelearningmastery.com/greedy-layer-wise-pretraining-tutorial/
import sklearn.model_selection
from sklearn.datasets import make_blobs
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import helperClass as help
import numpy as np


input_shape = (3, 3, 61)
# prepare the dataset
def prepare_data():
	x, y = help.load1Ddata(0)
	x, y = help.balanced_dataset(x, y, [1, 2, 3])
	return sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)


# define, fit and evaluate the base autoencoder
def base_autoencoder(trainX, testX):
	# define model
	model = Sequential()
	model.add(Dense(10, input_dim=61, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(61, activation='linear'))
	# compile model
	model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9))
	# fit model
	model.fit(trainX, trainX, epochs=10)
	# evaluate reconstruction loss
	train_mse = model.evaluate(trainX, trainX)
	test_mse = model.evaluate(testX, testX)
	print('> reconstruction error train=%.3f, test=%.3f' % (train_mse, test_mse))
	return model


# evaluate the autoencoder as a classifier
def evaluate_autoencoder_as_classifier(model, layers, trainX, trainy, testX, testy):
	# remember the current output layer
	#output_layer = model.layers[-1]
	for i in range(layers):
		# remove the output layer
		model.pop()
	# mark all remaining layers as non-trainable
	for layer in model.layers:
		layer.trainable = False
	# add new output layer
	model.add(Dense(4, activation='softmax'))
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])
	# fit model
	model.fit(x=trainX,
			y=trainy,
			epochs=10,
			validation_data=(testX, testy),
			batch_size=32)
# evaluate model
	_, train_acc = model.evaluate(trainX, trainy)
	_, test_acc = model.evaluate(testX, testy)
	# put the model back together
 #   model.pop()
 #   model.add(output_layer)
 #   model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9))
	return train_acc, test_acc


# add a model for supervised training
def add_model_to_autoencoder(model, layers, trainX, trainY, testX, testY):
	# remember the current output layer
	for i in range(layers):
		# remove the output layer
		model.pop()
	# mark all remaining layers as non-trainable
	for layer in model.layers:
		layer.trainable = False
	# add a new hidden layer
	model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
	#Todo: add layers
	model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])
	model.fit(x=trainX,
			y=trainY,
			epochs=10,
			validation_data=(testX, testY),
			batch_size=32)

#############################################unsupervised Pretraining############################################

# prepare unlabled data
x_ = np.load('C:/Users/Anna/Desktop/Masterarbeit/pkl/patient/1.pkl', allow_pickle=True)
x = x_.values[:, 2:63].astype(float)
for i in range(2,6):
	x_ = np.load('C:/Users/Anna/Desktop/Masterarbeit/pkl/patient/'+str(i) + '.pkl', allow_pickle=True)
	x = np.append(x, x_.values[:, 2:63].astype(float), axis=0)
	print(str(i) + ' done!')


trainX, testX = sklearn.model_selection.train_test_split(x, test_size=0.2, random_state=42)
model = base_autoencoder(trainX, testX)


trainX, testX, trainy, testy = prepare_data()
trainy = tf.keras.utils.to_categorical(trainy).astype(np.integer)
testy = tf.keras.utils.to_categorical(testy).astype(np.integer)
# get the base autoencoder
# evaluate the base model
scores = dict()
train_acc, test_acc = evaluate_autoencoder_as_classifier(model, 1, trainX, trainy, testX, testy)
print('> classifier accuracy layers=%d, train=%.3f, test=%.3f' % (len(model.layers), train_acc, test_acc))
# add layers and evaluate the updated model