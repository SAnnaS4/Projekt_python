import datetime
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import Learning.helperClass as help
import numpy as np

def get_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(20, kernel_size=5, activation='tanh',
                                     input_shape=(61, 1)))
    model.add(tf.keras.layers.MaxPool1D(pool_size = 3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='tanh'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

#x_, y_ = help.everaged1D()
x_, y_ = help.load1Ddata(1)
x, y = help.balanced_dataset(x_, y_, [0, 1])
x = np.reshape(x, (x.shape[0], 61, 1))
y = tf.keras.utils.to_categorical(y).astype(np.integer)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
print(x_train.shape)
model = get_model()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7)
# This callback will stop the training when there is no improvement in
# the validation loss for three consecutive epochs.
history = model.fit(x=x_train,
                      y=y_train,
                      epochs=100,
                      validation_data=(x_test, y_test),
                      callbacks=[callback],
                      batch_size=32)