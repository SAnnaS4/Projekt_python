# Load the TensorBoard notebook extension


#from tensorboard import program
#tb = program.TensorBoard()
#tb.configure(argv=[None, '--logdir', 'C:/Users/Anna/Desktop/Masterarbeit/logs'])
#url = tb.launch()

import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from LoadCSVDataLocation import LoadData
from sklearn.model_selection import GridSearchCV


import test_train_split_data

pathall = ['C:/Users/Anna/Desktop/Masterarbeit/data']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# LoadData(pathall, groupname0=0, groupname1=0, groupname2=1,groupname3=1, groupname4=2, groupname5=3).all_data()
EAC = pd.read_pickle('C:/Users/Anna/Desktop/Masterarbeit/data/a.pkl')
patientnumber = 94

print('test size of 10 %:' + str(patientnumber * 0.1))
print('##############################################################################################################')
testsize = round(patientnumber * 0.1)
print('Patient Number: ' + str(patientnumber))

i = 40
patients = np.array(EAC['patients'])
patients = np.int_(patients)
datastest = EAC.values[patients >= i, 3:]
labelstest = EAC.values[patients >= i, 0]
Patienttestpath = EAC.values[patients >= i, 2]
#
EAC_test = pd.DataFrame(datastest)
EAC_test.insert(loc=0, column='label', value=labelstest)

EAC_test.columns = ['label', '500nm', '505nm', '510nm', '515nm', '520nm',
                    '525nm', '530nm', '535nm', '540nm',
                    '545nm', '550nm', '555nm', '560nm', '565nm', '570nm', '575nm',
                    '580nm', '585nm', '590nm',
                    '595nm',
                    '600nm', '605nm', '610nm', '615nm', '620nm', '625nm', '630nm',
                    '635nm', '640nm', '645nm',
                    '650nm', '655nm', '660nm', '665nm', '670nm', '675nm', '680nm',
                    '685nm', '690nm', '695nm',
                    '700nm', '705nm', '710nm', '715nm', '720nm', '725nm', '730nm',
                    '735nm', '740nm', '745nm',
                    '750nm', '755nm', '760nm', '765nm', '770nm', '775nm', '780nm',
                    '785nm', '790nm', '795nm',
                    '800nm', 'c1', 'c2', 'c3', 'c4']
#c1, c2, c3, c4 raus

# 65 + lables
print('Patienttestset')
print("EAC - ", 1)
print("Plattenepithel - ", 2)
print("Stroma - ", 3)
print("Blank - ", 4)

labels = np.array(EAC_test['label'])
labels = np.int_(labels)
print(labels)
x_train, y_train, x_test, y_test = test_train_split_data.train_test(
    EAC_test).div_set_balanced_3classes()

def showSomeData(x_test, y_test):
    eac = x_test[np.where(y_test == 1)[0][0]]
    eac_mean = x_test[np.where(y_test == 1)].mean(axis=0)
    eac_std = x_test[np.where(y_test == 1)].std(axis=0)
    pe = x_test[np.where(y_test == 2)[0][0]]
    pe_mean = x_test[np.where(y_test == 2)].mean(axis=0)
    pe_std = x_test[np.where(y_test == 2)].std(axis=0)
    stroma = x_test[np.where(y_test == 3)[0][0]]
    stroma_mean = x_test[np.where(y_test == 3)].mean(axis=0)
    stroma_std = x_test[np.where(y_test == 3)].std(axis=0)
    blank = x_test[np.where(y_test == 4)[0][0]]
    blank_mean = x_test[np.where(y_test == 4)].mean(axis=0)
    blank_std = x_test[np.where(y_test == 4)].std(axis=0)

    fig, axs = plt.subplots(4, 3, figsize=(10, 10))
    axs[0, 0].plot(eac)
    axs[0, 1].plot(eac_mean)
    axs[0, 2].plot(eac_std)
    axs[1, 0].plot(pe)
    axs[1, 1].plot(pe_mean)
    axs[1, 2].plot(pe_std)
    axs[2, 0].plot(stroma)
    axs[2, 1].plot(stroma_mean)
    axs[2, 2].plot(stroma_std)
    axs[3, 0].plot(blank)
    axs[3, 1].plot(blank_mean)
    axs[3, 2].plot(blank_std)
    plt.show()

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

def get_model(layer_sizes, dropout, learning_rate, activation):
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(layer_sizes[0], activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    for dense_size in layer_sizes[1:]:
        model.add(tf.keras.layers.Dense(dense_size, activation=activation))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    # opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    print("{0}, {1}, {2}, {3}".format(layer_sizes, dropout, learning_rate, activation))
    return model


layer_sizes = [(128, 128, 128), (128, 128, 64, 64), (
128, 128, 64, 32)]  # , (64, 32), (16,)]#,(32,), (16, 16), (32, 32), (64,), (16,16,16)]#, [16, 16], [32, 32]]
classifier = tf.keras.wrappers.scikit_learn.KerasClassifier(get_model, batch_size=32)
validator = GridSearchCV(classifier,
                         param_grid={'epochs': [40],  # , 20, 30],
                                     'layer_sizes': layer_sizes,
                                     'dropout': [0.2],  # [x * 0.1 for x in range(2, 3)],
                                     'learning_rate': [0.001],
                                     'activation': ['relu']}, n_jobs=1)


# log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# validator.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
# validator.fit(x_train, y_train)
# print(validator.best_params_)
# print(validator.best_score_)

def mlp(x_train, y_train, x_test, y_test):
    ##Modell
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))

    ##Training
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('##################################Fitting######################################')

   # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(x=x_train, y=y_train, epochs=60)
    print('##################################Evaluierung##################################')
    print('EAC')
    model.evaluate(x_test[np.where(y_test == 1)], y_test[np.where(y_test == 1)])
    print('Plattenepithel')
    model.evaluate(x_test[np.where(y_test == 2)], y_test[np.where(y_test == 2)])
    print('Stroma')
    model.evaluate(x_test[np.where(y_test == 3)], y_test[np.where(y_test == 3)])
    print('blank')
    model.evaluate(x_test[np.where(y_test == 4)], y_test[np.where(y_test == 4)])
    return model

showSomeData(x_train, y_train)
kmean(x_train, y_train, x_test, y_test)
model_mlp = mlp(x_train,y_train,x_test,y_test)
model_mlp.save_weights('C:/Users/Anna/Desktop/Masterarbeit/checkpoints/myCheckpoint')
# model.load_weights('C:/Users/Anna/Desktop/Masterarbeit/checkpoints/myCheckpoint')

#größe Modell Marianne schicken
#pixel abschnitte zusammen betrachten
#teste testgrößen