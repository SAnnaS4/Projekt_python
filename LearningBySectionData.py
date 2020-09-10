import sklearn.model_selection
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.python.keras
import tensorflow.python.keras.utils.np_utils
import os

from sklearn.model_selection import train_test_split

config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

npzpath = 'C:/Users/Anna/Desktop/Masterarbeit/npz'

file_list = os.listdir(npzpath)
loaded = np.load('C:/Users/Anna/Desktop/Masterarbeit/npz/eac1.npz')
x = loaded['x']
y = np.array(loaded['y'][:, 0]).astype(np.integer) - 1

for file in file_list:
    if not file.endswith('eac1.npz'):
        name = npzpath + '/' + file
        x_ = np.load(name)['x']
        y_ = np.array(np.load(name)['y'][:, 0]).astype(np.integer) - 1
        x = np.append(x, x_, axis=0)
        y = np.append(y, y_, axis=0)

# 1 => EAC
# 2 => Stroma
# 3 => Plattenepithel
# 4 => Blank


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# -- Preparatory code --
# Model configuration
batch_size = 10
nr_epochs = 10
learning_rate = 0.002
no_classes = 4
validation_split = 0.2
verbosity = 1

# Determine sample shape
# sample_shape = (61, 3, 3, 1)
# x_train = x_train.reshape(x_train.shape[0], 61, 3, 3, 1)
# x_test = x_test.reshape(x_test.shape[0], 61, 3, 3, 1)


# Convert target vectors to categorical targets
targets_train = tensorflow.python.keras.utils.np_utils.to_categorical(y_train).astype(np.integer)
targets_test = tensorflow.keras.utils.to_categorical(y_test).astype(np.integer)

# Create the model as 3D
# model = tensorflow.keras.Sequential()
# model.add(
#     tensorflow.python.keras.layers.Conv3D(32, kernel_size=(1, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
# model.add(tensorflow.keras.layers.MaxPooling3D(pool_size=(61, 1, 1)))
# model.add(tensorflow.keras.layers.Dropout(0.5))
# model.add(tensorflow.keras.layers.Flatten())
# model.add(tensorflow.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
# model.add(tensorflow.keras.layers.Dense(no_classes, activation='softmax'))

sample_shape = (61, 3, 3)
x_train = x_train.reshape(x_train.shape[0], 61, 3, 3)
x_test = x_test.reshape(x_test.shape[0], 61, 3, 3)

depth_pool_mean = tensorflow.keras.layers.Lambda(
    lambda X: tf.compat.v1.reduce_mean(X, axis=[3], keepdims=True))

depth_pool_max = tensorflow.keras.layers.Lambda(
    lambda X: tf.compat.v1.reduce_max(X, axis=[3], keepdims=True))

#Create model as 2D + 61 channel
# model = tensorflow.keras.Sequential()
# model.add(tensorflow.keras.layers.Conv2D(128, kernel_size=3, activation='relu', kernel_initializer='he_uniform',
#                                          input_shape=sample_shape, padding="SAME"))
# model.add(depth_pool_max)
# model.add(tensorflow.keras.layers.Conv2D(128, kernel_size=3, activation='relu', kernel_initializer='he_uniform',
#                                          padding="SAME"))
# model.add(depth_pool_max)
# model.add(tensorflow.keras.layers.Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='he_uniform',
#                                          padding="SAME"))
# model.add(depth_pool_max)
# model.add(tensorflow.keras.layers.Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='he_uniform',
#                                          padding="SAME"))
# model.add(depth_pool_max)
# model.add(tensorflow.keras.layers.Flatten())
# model.add(tensorflow.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
# model.add(tensorflow.keras.layers.Dense(no_classes, activation='softmax'))
#
# # Compile the model
# model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
#               optimizer=tensorflow.keras.optimizers.Adam(lr=learning_rate),
#               metrics=['accuracy'])
# model.summary()
#
# # Fit data to model
# history = model.fit(x_train, targets_train,
#                     batch_size=batch_size,
#                     epochs=nr_epochs,
#                     verbose=verbosity,
#                     validation_split=validation_split)
#
# # Generate generalization metrics
# score = model.evaluate(x_test, targets_test, verbose=0)
# print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
#
# # Plot history: Categorical crossentropy & Accuracy
# plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
# plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
# plt.plot(history.history['accuracy'], label='Accuracy (training data)')
# plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
# plt.title('Model performance')
# plt.ylabel('Loss value')
# plt.xlabel('N. epoch')
# plt.legend(loc="upper left")
# plt.show()

from tensorboard import program
from _datetime import datetime

#tb = program.TensorBoard()
#tb.configure(argv=[None, '--logdir', 'C:/Users/Anna/Desktop/Masterarbeit/logs'])
#url = tb.launch()


def get_model(layer_sizes, kernel_size, kernel_initializer, depth_pool, dropout, learning_rate, activation):
    model = tf.keras.Sequential()
    i = 0
    model.add(tensorflow.keras.layers.Conv2D(layer_sizes[0], kernel_size=kernel_size, activation=activation,
                                             input_shape=sample_shape, kernel_initializer=kernel_initializer,
                                             padding="SAME"))
    model.add(depth_pool_max)
    i = ++i
    for layer in layer_sizes[1:]:
        model.add(tensorflow.keras.layers.Conv2D(layer, kernel_size=kernel_size, activation=activation,
                                                 kernel_initializer=kernel_initializer, padding="SAME"))
        model.add(depth_pool_max)
        i = ++i
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tensorflow.keras.layers.Flatten())
    model.add(tensorflow.keras.layers.Dense(256, activation=activation, kernel_initializer='he_uniform'))
    model.add(tensorflow.keras.layers.Dense(no_classes, activation='softmax'))
    # opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print("{0}, {1}, {2}, {3}, {4}, {5}, {6}".format(layer_sizes, kernel_size, kernel_initializer, depth_pool.__name__, dropout,
                                                     learning_rate, activation))
    return model


layer_sizes = [(128, 128, 128), (128, 128, 64, 64), (128, 128, 64, 32), (16, 16, 16, 16), (64, 32, 16)]
depth_pool = [(depth_pool_max, depth_pool_max, depth_pool_max, depth_pool_max),
              (depth_pool_mean, depth_pool_mean, depth_pool_mean, depth_pool_mean),
              (depth_pool_max, depth_pool_mean, depth_pool_max, depth_pool_mean)]
classifier = tf.keras.wrappers.scikit_learn.KerasClassifier(get_model, batch_size=10)
validator = sklearn.model_selection.GridSearchCV(classifier,
                                                 param_grid={'epochs': [40],  # , 20, 30],
                                                             'kernel_size': [2, 3],
                                                             'kernel_initializer': ['he_uniform'],
                                                             'depth_pool': depth_pool,
                                                             'layer_sizes': layer_sizes,
                                                             'dropout': [x * 0.1 for x in range(0, 2)],
                                                             'learning_rate': [0.001, 0.01],
                                                             'activation': ['relu']}, n_jobs=1)

#log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#validator.fit(x_train, targets_train, validation_data=(x_test, targets_test), callbacks=[tensorboard_callback])
validator.fit(x_train, y_train)
print(validator.best_params_)
print(validator.best_score_)
