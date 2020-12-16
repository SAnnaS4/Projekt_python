import datetime
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import Learning.helperClass as help
import numpy as np
from tensorflow.keras.layers import TimeDistributed

def get_model():
    model = tf.keras.Sequential()
    model.add(TimeDistributed(tf.keras.layers.Conv1D(16, kernel_size=3, activation='tanh', input_shape=(6, 1))))
   # model.add(TimeDistributed(tf.keras.layers.MaxPool1D(pool_size = 3)))
    model.add(TimeDistributed(tf.keras.layers.Dropout(0.1)))
    model.add(TimeDistributed(tf.keras.layers.Conv1D(16, kernel_size=3, activation='tanh')))
    #model.add(TimeDistributed(tf.keras.layers.MaxPool1D(pool_size=3)))
    model.add(TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.SimpleRNN(512, unroll=True, return_sequences = True, return_state=False))
    model.add(tf.keras.layers.SimpleRNN(256, unroll=True))
   # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(100, activation='tanh'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    learning_rate = 0.01
    decay_rate = learning_rate / 30
    momentum = 0.8
    opt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

x_, y_ = help.load1Ddata(0)
x, y = help.balanced_dataset(x_, y_, [1, 2, 3])
x = x[:, 0:60]
n_steps, n_length = 10, 6
x = x.reshape((x.shape[0], n_steps, n_length, 1))
y = tf.keras.utils.to_categorical(y).astype(np.integer)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
print(x_train.shape)
model = get_model()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7)

history = model.fit(x=x_train,
                      y=y_train,
                      epochs=100,
                      validation_data=(x_test, y_test),
                      callbacks=[callback],
                      batch_size=32)
print(len(history.history['loss']))