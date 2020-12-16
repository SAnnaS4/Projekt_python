import numpy
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

# for i in range(np.size(y)):
#     if y[i] == 2:
#         y[i] = 1
#     else:
#         y[i] = 0

x = x[y<3]
y = y[y<3]


# Convert target vectors to categorical targets
targets_train = tensorflow.keras.utils.to_categorical(y).astype(np.integer)

def loadModel_custPooling(number):
    modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/balanced/hparam/20201128-235319run-" + str(number) + ".h5"
    model = tensorflow.keras.models.load_model(modelPath, custom_objects={'CustomizedPooling': cp.CustomizedPooling})
    return model

def loadModel_3D(number):
    modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/balanced/hparam3D/20201201-054803run-" + str(number) + ".h5"
    model = tensorflow.keras.models.load_model(modelPath)
    return model

def loadModel_bin_custPooling(number):
    modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/hparam/20201108-094422run-" + str(number) + ".h5"
    model = tensorflow.keras.models.load_model(modelPath, custom_objects={'CustomizedPooling': cp.CustomizedPooling})
    return model

class Depth_pool_selection(tf.keras.layers.Layer):
    def __init__(self, reduce_function=1, **kwargs):
        self.reduce_function = reduce_function
        super(Depth_pool_selection, self).__init__(**kwargs)

    def call(self, input_data, **kwargs):
        if self.reduce_function:
            return tf.compat.v1.reduce_mean(input_data, axis=[3], keepdims=True)
        else:
            return tf.compat.v1.reduce_max(input_data, axis=[3], keepdims=True)

    def get_config(self):
        config = super().get_config().copy()
        return config

# nochmal mit custPooling
def loadModel_turned(number):
    modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/balanced/hparamT/20201128-112727run-" + str(number) + ".h5"
    model = tensorflow.keras.models.load_model(modelPath,
                                               custom_objects={'Depth_pool_selection': Depth_pool_selection})
    return model


def loadModel_poolingy(number):
    modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/balanced/hparam1/20201130-132548run-" + str(number) + ".h5"
    model = tensorflow.keras.models.load_model(modelPath, custom_objects={'Depth_pool_selection': Depth_pool_selection})
    return model


#model = loadModel_custPooling(44)
# model = loadModel_bin_custPooling(12)
#model = loadModel_poolingy(36)
#sample_shape = (3, 3, 61)
#x = x.reshape(x.shape[0], 3, 3, 61)

#model = loadModel_turned(36)
#sample_shape = (61, 3, 3)
#x = x.reshape(x.shape[0], 61, 3, 3)

model = loadModel_3D(27)
sample_shape = (3, 3, 61, 1)
x = x.reshape(x.shape[0], 3, 3, 61, 1)

print("Model loaded")

eac = x[np.where(y == 0)]
stroma = x[np.where(y == 1)]
plattenep = x[np.where(y == 2)]
blank = x[np.where(y == 3)]

print("eac: " + str(np.size(eac, axis=0)))
print("stroma: " + str(np.size(stroma, axis=0)))
print("plattenep: " + str(np.size(plattenep, axis=0)))
print("blank: " + str(np.size(blank, axis=0)))


print('##################################Evaluierung##################################')
y_pred_class = model.predict_classes(x)

from sklearn import metrics
from itertools import cycle

###################################ROC-Curve##############################################
n_classes = 3
lw = 2
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    helper_1 = np.zeros(y.shape)
    helper_2 = np.zeros(y.shape)
    fpr[i], tpr[i], _ = metrics.roc_curve(np.where(y == i, 1, helper_1), np.where(y_pred_class == i, 1, helper_2))
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += numpy.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


###################################Precision-Recall############################################
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    helper_1 = np.where(y == i, 1, np.zeros(y.shape))
    helper_2 = np.where(y_pred_class == i, 1, np.zeros(y_pred_class.shape))
    precision[i], recall[i], _ = precision_recall_curve(helper_1,helper_2)
    average_precision[i] = average_precision_score(helper_1, helper_2)

colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    ax_y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[ax_y >= 0], ax_y[ax_y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, ax_y[45] + 0.02))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
plt.show()
###################################others############################################

for i in range(n_classes):
    print("################# Class: " + str(i) + " ######################")
    helper_1 = np.where(y == i, 1, np.zeros(y.shape))
    helper_2 = np.where(y_pred_class == i, 1, np.zeros(y.shape))
    tn, fp, fn, tp = metrics.confusion_matrix(helper_1, helper_2).ravel()
    print('MCC: ' + str(metrics.matthews_corrcoef(helper_1, helper_2)))
    print('F1-Score: ' + str(metrics.f1_score(helper_1, helper_2)))
    print('SENS: ' + str(metrics.recall_score(helper_1, helper_2)))
    print('SPEC: ' + str(tn / (tn + fp)))
