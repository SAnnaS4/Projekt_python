# Load the TensorBoard notebook extension


# from tensorboard import program
# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', 'C:/Users/Anna/Desktop/Masterarbeit/logs'])
# url = tb.launch()
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

import data_analyses
from LoadCSVDataLocation import LoadData
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import Learning.helperClass as help

datastest, labelstest = help.load1Ddata(1)
datastest, labelstest = help.balanced_dataset(datastest, labelstest, [0, 1])
x_train, x_test, y_train, y_test = train_test_split(datastest, labelstest, test_size=0.1, random_state=42)

def kmean(x_train, y_train, x_test, y_test):
    # mit mehr Daten könnte es gut werden --> alle Bilder verwenden
    from sklearn.cluster import KMeans, SpectralClustering
    kmeans = KMeans(n_clusters=4, random_state=1).fit(x_train)
    prediction = kmeans.predict(x_test[np.where(y_test == 1)])
    pred_vollst = kmeans.predict(x_test)

    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    axs[0].plot(x_test[np.where(pred_vollst == 0)].mean(axis=0))
    axs[1].plot(x_test[np.where(pred_vollst == 1)].mean(axis=0))
    axs[2].plot(x_test[np.where(pred_vollst == 2)].mean(axis=0))
    axs[3].plot(x_test[np.where(pred_vollst == 3)].mean(axis=0))
    plt.show()

    print("#############EAC#############")
    print("Anzahl 0", np.where(prediction == 0)[0].size)
    print("Anzahl 1", np.where(prediction == 1)[0].size)
    print("Anzahl 2", np.where(prediction == 2)[0].size)
    print("Anzahl 3", np.where(prediction == 3)[0].size)
    prediction = kmeans.predict(x_test[np.where(y_test == 2)])
    print("################Plattenepithel###########")
    print("Anzahl 0", np.where(prediction == 0)[0].size)
    print("Anzahl 1", np.where(prediction == 1)[0].size)
    print("Anzahl 2", np.where(prediction == 2)[0].size)
    print("Anzahl 3", np.where(prediction == 3)[0].size)
    prediction = kmeans.predict(x_test[np.where(y_test == 3)])
    print("################Stroma###########")
    print("Anzahl 0", np.where(prediction == 0)[0].size)
    print("Anzahl 1", np.where(prediction == 1)[0].size)
    print("Anzahl 2", np.where(prediction == 2)[0].size)
    print("Anzahl 3", np.where(prediction == 3)[0].size)
    prediction = kmeans.predict(x_test[np.where(y_test == 4)])
    print("################Blank###########")
    print("Anzahl 0", np.where(prediction == 0)[0].size)
    print("Anzahl 1", np.where(prediction == 1)[0].size)
    print("Anzahl 2", np.where(prediction == 2)[0].size)
    print("Anzahl 3", np.where(prediction == 3)[0].size)



def mlp(x_train, y_train, x_test, y_test):
    log_dir = 'C:/Users/Anna/Desktop/Masterarbeit/logs/blank_class'
    ##Modell
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
    #model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

    ##Training
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('##################################Fitting######################################')

    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(x=x_train, y=y_train, epochs=10)
    print('##################################Evaluierung##################################')
    print('notBlank')
    model.evaluate(x_test[np.where(y_test == 0)], y_test[np.where(y_test == 0)])
    print('blank')
    model.evaluate(x_test[np.where(y_test == 1)], y_test[np.where(y_test == 1)])
    model.save(log_dir + ".h5")
    return model

#x_test = x_test.astype(np.float)
#x_train = x_train.astype(np.float)
data_analyses.showSomeData(x_test, y_test)
#kmean(x_train, y_train, x_test, y_test)
model_mlp = mlp(x_train, y_train, x_test, y_test)
#model_mlp.save_weights('C:/Users/Anna/Desktop/Masterarbeit/checkpoints/myCheckpoint')
# model.load_weights('C:/Users/Anna/Desktop/Masterarbeit/checkpoints/myCheckpoint')
model_blank = tf.keras.models.load_model('C:/Users/Anna/Desktop/Masterarbeit/logs/blank_class.h5')

model_blank.evaluate(x_train[np.where(y_train == 1)], y_train[np.where(y_train == 1)])