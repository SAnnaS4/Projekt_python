import sklearn.model_selection
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.python.keras
import tensorflow.python.keras.utils.np_utils
import os

from sklearn.model_selection import train_test_split

#config = tf.compat.v1.ConfigProto(gpu_options=
#                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
#                                  )
#config.gpu_options.allow_growth = True

#session = tf.compat.v1.Session(config=config)
#tf.compat.v1.keras.backend.set_session(session)

npzpath = 'C:/Users/Anna/Desktop/Masterarbeit/npz'
#npzpath = '/home/sc.uni-leipzig.de/ay312doty/npz'

file_list = os.listdir(npzpath)
loaded = np.load(npzpath + '/eac1.npz')
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
sample_shape = (61, 3, 3, 1)
x_train = x_train.reshape(x_train.shape[0], 61, 3, 3, 1)
x_test = x_test.reshape(x_test.shape[0], 61, 3, 3, 1)


# Convert target vectors to categorical targets
targets_train = tensorflow.python.keras.utils.np_utils.to_categorical(y_train).astype(np.integer)
targets_test = tensorflow.keras.utils.to_categorical(y_test).astype(np.integer)

# Create the model as 3D
model = tensorflow.keras.Sequential()
model.add(
    tensorflow.keras.layers.Conv3D(32, kernel_size=(1, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
model.add(tensorflow.keras.layers.MaxPooling3D(pool_size=(61, 1, 1)))
model.add(tensorflow.keras.layers.Dropout(0.5))
model.add(tensorflow.keras.layers.Flatten())
model.add(tensorflow.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(tensorflow.keras.layers.Dense(no_classes, activation='softmax'))

# Compile the model
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
model.summary()

# Fit data to model
history = model.fit(x_train, targets_train,
                    batch_size=batch_size,
                    epochs=nr_epochs,
                    verbose=verbosity,
                    validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(x_test, targets_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Plot history: Categorical crossentropy & Accuracy
plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
plt.plot(history.history['accuracy'], label='Accuracy (training data)')
plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
plt.title('Model performance')
plt.ylabel('Loss value')
plt.xlabel('N. epoch')
plt.legend(loc="upper left")
plt.show()