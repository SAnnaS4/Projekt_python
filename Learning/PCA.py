import Learning.helperClass as help
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Learning.customizedPooling import CustomizedPooling
import Learning.kFold_training as kfold

x_pca, y_pca = help.load1Ddata(0)
datastest, labelstest = help.balanced_dataset(x_pca, y_pca, [1, 2, 3])
x_pca_train, x_pca_test, y_pca_train, y_pca_test = train_test_split(datastest, labelstest, test_size=0.1,
                                                                    random_state=42)

# PCA-Test
pcas = np.array([6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61])
x, y = help.load3Ddata()
eind = np.reshape(x, (x.shape[0] * 9, 61))


def pcaFor3DdataSet(pca):
    x_train1d = pca.transform(eind)
    x_training = np.reshape(x_train1d, (x.shape[0], 3, 3, x_train1d.shape[1]))
    return x_training, y


def get_model_CNN2d(layer_sizes, sample_shape):
    model = tf.keras.Sequential()
    i = 1
    model.add(tf.keras.layers.Conv2D(layer_sizes[0], kernel_size=3, activation='relu',
                                     input_shape=sample_shape, kernel_initializer='he_uniform',
                                     padding="SAME"))
    model.add(CustomizedPooling(reduce_function=1))
    for layer in layer_sizes[1:]:
        model.add(tf.keras.layers.Conv2D(layer, kernel_size=3, activation='relu',
                                         kernel_initializer='he_uniform', padding="SAME"))
        model.add(CustomizedPooling(reduce_function=1))
        i = ++i
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    metrics = ['accuracy']
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)
    model.summary()
    return model


def get_model_CNN3D(sample_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(128, kernel_size=(1, 1, 3), activation='relu', kernel_initializer='he_uniform',
                                     input_shape=sample_shape))
    if (1):
        model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 1, 3)))
    for i in range(3):
        model.add(
            tf.keras.layers.Conv3D(128, kernel_size=(1, 1, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 1, 3)))
        model.add(tf.keras.layers.Dropout(0.01))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    metrics = ['accuracy']
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
    model.summary()
    # fit (Learningrate 0.0010000)
    return model


def get_model_CNN1D(sample_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(20, kernel_size=5, activation='tanh',
                                     input_shape=sample_shape))
    model.add(tf.keras.layers.MaxPool1D(pool_size=3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='tanh'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


epochs = 15
folds = 10
median_values = np.zeros((0, folds, epochs))
for i in pcas:
    print("Komponenten: " + str(i))
    pca = PCA(n_components=i)
    pca.fit(x_pca_train)

    x_, y_ = pcaFor3DdataSet(pca)
    #x_, y_ = help.everaged1D()
    #x_ = np.reshape(x_, (x_.shape[0], 61, 1))

    datastest, labelstest = help.balanced_dataset(x_, y_, [0, 2, 3])
    targets_train = tf.keras.utils.to_categorical(labelstest).astype(np.integer)

    model = get_model_CNN2d((16, 16, 16, 16), (x_.shape[1], x_.shape[2], x_.shape[3]))
    #model = get_model_CNN1D((x_.shape[1], x_.shape[2]))

    histories = kfold.train(datastest, targets_train, model, folds, epochs)
    median_values = np.append(median_values, np.reshape(histories, (1, folds, epochs)), axis=0)
path = "C:/Users/Anna/Desktop/Masterarbeit/data/pcaValues2D"
np.savez_compressed(path, median_values=median_values)

# x_train, x_test, y_train, y_test = train_test_split(datastest, targets_train, test_size=0.2,
#                                                                     random_state=42)

# callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7)
# model.fit(x=x_train,
#           y=y_train,
#           epochs=50,
#           validation_data=(x_test, y_test),
#           callbacks=[callback],
#           batch_size=32)

# mit k-fold evaluieren (10) + 1D CNN mgw auf 3D + mgw ohne blank versuchen
# + darstellung der Ergebnisse als Balkendiagram
