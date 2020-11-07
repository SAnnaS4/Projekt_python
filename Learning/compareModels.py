import sklearn.model_selection
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.python.keras
import tensorflow.python.keras.layers
import tensorflow.python.keras.models
import tensorflow.python.keras.utils.np_utils
import os
import Learning.customizedPooling as cp

npzpath = 'C:/Users/Anna/Desktop/Masterarbeit/npz'

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

# Determine sample shape
sample_shape = (3, 3, 61)
x = x.reshape(x.shape[0], 3, 3, 61)

# Convert target vectors to categorical targets
targets_train = tensorflow.keras.utils.to_categorical(y).astype(np.integer)

def loadModel_custPooling(number):
    modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/test/hparam/20201023-105910run-" + str(number) + ".h5"
    model = tensorflow.keras.models.load_model(modelPath, custom_objects={'CustomizedPooling': cp.CustomizedPooling})
    return model

def loadModel_bin_custPooling(number):
    modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/binär/hparam/20201029-081317run-" + str(number) + ".h5"
    model = tensorflow.keras.models.load_model(modelPath, custom_objects={'CustomizedPooling': cp.CustomizedPooling})
    return model

depth_pool_mean = tensorflow.keras.layers.Lambda(
    lambda X: tf.compat.v1.reduce_mean(X, axis=[3], keepdims=True))

depth_pool_max = tensorflow.keras.layers.Lambda(
    lambda X: tf.compat.v1.reduce_max(X, axis=[3], keepdims=True))

def depth_pool_selection(i):
    if i == 0: return depth_pool_max
    else: return depth_pool_mean

#nochmal mit custPooling
def loadModel_turned(number):
    modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/mitModel/hparam/20201012-114553run-" + str(number) + ".h5"
    model = tensorflow.keras.models.load_model(modelPath,
                                               custom_objects={'lambda': depth_pool_max,
                                                               })
    return model

def loadModel_poolingy(number):
    modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/richtigtum/hparam/20201019-122436run-" + str(number) + ".h5"
    model = tensorflow.keras.models.load_model(modelPath, custom_objects={'lambda': depth_pool_mean,
                                                                          })
    return model

model = loadModel_custPooling(44)
model1 = loadModel_bin_custPooling(7)
print("Model loaded")

eac = x[np.where(y== 0)]
stroma = x[np.where(y== 1)]
plattenep = x[np.where(y== 2)]
blank = x[np.where(y== 3)]

print('##################################Evaluierung##################################')
model.evaluate(x, targets_train)
# print('EAC: ' + str(np.size(eac)))
# model.evaluate(x[np.where(y == 0)], targets_train[np.where(y == 0)])
# print('Plattenepithel: ' + str(np.size(plattenep)))
# model.evaluate(x[np.where(y == 2)], targets_train[np.where(y == 2)])
# print('Stroma: ' + str(np.size(stroma)))
# model.evaluate(x[np.where(y == 1)], targets_train[np.where(y == 1)])
# print('blank: ' + str(np.size(blank)))
# model.evaluate(x[np.where(y == 3)], targets_train[np.where(y == 3)])

y_pred_class = model.predict_classes(x)

from sklearn import metrics

for i in range(np.size(y)):
    if y_pred_class[i] == 2:
        y_pred_class[i] = 1
    else:
        y_pred_class[i] = 0

    if y[i] == 2:
        y[i] = 1
    else:
        y[i] = 0


tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred_class).ravel()
print('MCC: ' + str(metrics.matthews_corrcoef(y, y_pred_class)))
print('F1-Score: ' + str(metrics.f1_score(y, y_pred_class)))
print('SENS: ' + str(metrics.recall_score(y, y_pred_class)))
print('SPEC: ' + str(tn / (tn + fp)))
print('ROC: ' + str(metrics.roc_auc_score(y, y_pred_class)))
print('Precision-Recall: ' + str(metrics.average_precision_score(y, y_pred_class,  average='micro')))


#print('andere: ' + str(np.size(eac)))
#model.evaluate(x[np.where(y == 0)], targets_train[np.where(y == 0)])
#print('Plattenepithel: ' + str(np.size(stroma)))
#model.evaluate(x[np.where(y == 1)], targets_train[np.where(y == 1)])
