from sklearn.model_selection import KFold
import numpy as np


def train(x, y, model, folds, epochs, callback = 0, patience = 0):
    x, y, model = x, y, model
    kf = KFold(n_splits=folds)
    list_of_history = np.array([])
    for train_index, test_index in kf.split(y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if(patience):
            history = model.fit(x=x_train,
                              y=y_train,
                              epochs=100,
                              validation_data=(x_test, y_test),
                              callbacks = [callback],
                              batch_size=32)
        else:
            #history.history: pro epoch alle metriken (val_accuracy = [0.5, 0.6..])
           history = model.fit(x=x_train,
                              y=y_train,
                              epochs=epochs,
                              validation_data=(x_test, y_test),
                              batch_size=32)
        if list_of_history.size == 0:
            list_of_history = np.asarray(history.history['val_accuracy'])
            list_of_history = np.reshape(list_of_history, (1, epochs))
        else:
            neu = np.reshape((np.asarray(history.history['val_accuracy'])), (1, epochs))
            list_of_history = np.append(list_of_history, neu, axis=0)
    return list_of_history




























































