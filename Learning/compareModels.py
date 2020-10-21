import sklearn.model_selection
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.python.keras
import tensorflow.python.keras.layers
import tensorflow.python.keras.models
import tensorflow.python.keras.utils.np_utils
import os

npzpath = 'C:/Users/Anna/Desktop/Masterarbeit/npz'
#npzpath = '/home/sc.uni-leipzig.de/ay312doty/npz'

file_list = os.listdir(npzpath)
loaded = np.load(npzpath + '/eac1.npz')
x = loaded['x']
y = np.array(loaded['y'][:, 0]).astype(np.integer) - 1

# for file in file_list:
#     if not file.endswith('eac1.npz'):
#         name = npzpath + '/' + file
#         x_ = np.load(name)['x']
#         y_ = np.array(np.load(name)['y'][:, 0]).astype(np.integer) - 1
#         x = np.append(x, x_, axis=0)
#         y = np.append(y, y_, axis=0)

# 1 => EAC
# 2 => Stroma
# 3 => Plattenepithel
# 4 => Blank

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
x = x.reshape(x.shape[0], 61, 3, 3, 1)

# Convert target vectors to categorical targets
targets_train = tensorflow.keras.utils.to_categorical(y).astype(np.integer)

modelPath = "C:/Users/Anna/Desktop/Masterarbeit/SlurmLog/mitModel/hparam/20201012-114553run-28.h5"
model = tensorflow.keras.models.load_model(modelPath)

print("Model loaded")

plt.plot(model.history['loss'], label='Categorical crossentropy (training data)')
plt.plot(model.history['val_loss'], label='Categorical crossentropy (validation data)')
plt.plot(model.history['accuracy'], label='Accuracy (training data)')
plt.plot(model.history['val_accuracy'], label='Accuracy (validation data)')
plt.title('Model performance')
plt.ylabel('Loss value')
plt.xlabel('N. epoch')
plt.legend(loc="upper left")
plt.show()

eac = x[np.where(y= 1)]
stroma = x[np.where(y= 2)]
plattenep = x[np.where(y= 3)]
blank = x[np.where(y= 4)]

print('##################################Evaluierung##################################')
print('EAC: ' + np.size(eac))
model.evaluate(x[np.where(y == 1)], y[np.where(y == 1)])
print('Plattenepithel: ' + np.size(plattenep))
model.evaluate(x[np.where(y == 3)], y[np.where(y == 3)])
print('Stroma: ' + np.size(stroma))
model.evaluate(x[np.where(y == 2)], y[np.where(y == 2)])
print('blank: ' + np.size(blank))
model.evaluate(x[np.where(y == 4)], y[np.where(y == 4)])
