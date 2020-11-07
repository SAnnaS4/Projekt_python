import sklearn.model_selection
import tensorflow as tf
import numpy as np
import tensorflow.python.keras
import tensorflow.python.keras.utils.np_utils
import os

#import tensorflow.python.layers.base
from sklearn.model_selection import train_test_split

import Learning.CNN2d

config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  )
config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

npzpath = 'C:/Users/Anna/Desktop/Masterarbeit/npz'
#npzpath = '/home/sc.uni-leipzig.de/ay312doty/npz'

file_list = os.listdir(npzpath)
loaded = np.load(npzpath + '/eac1.npz')
x = loaded['x']
y = np.array(loaded['y'][:, 0]).astype(np.integer) - 1

#for file in file_list:
#    if not file.endswith('eac1.npz'):
#        name = npzpath + '/' + file
#        x_ = np.load(name)['x']
#        y_ = np.array(np.load(name)['y'][:, 0]).astype(np.integer) - 1
#        x = np.append(x, x_, axis=0)
#        y = np.append(y, y_, axis=0)

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


# Convert target vectors to categorical targets
targets_train = tensorflow.keras.utils.to_categorical(y_train).astype(np.integer)
targets_test = tensorflow.keras.utils.to_categorical(y_test).astype(np.integer)

sample_shape = (61, 3, 3)
x_train = x_train.reshape(x_train.shape[0], 61, 3, 3)
x_test = x_test.reshape(x_test.shape[0], 61, 3, 3)


########################################CustomizedPooling Layer################################################
class Depth_pool_selection(tf.keras.layers.Layer):
    def __init__(self, reduce_function=1,  **kwargs):
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

########################################Lambda Pooling Layer################################################

depth_pool_mean = tensorflow.keras.layers.Lambda(
    lambda X: tf.compat.v1.reduce_mean(X, axis=[3], keepdims=True))

depth_pool_max = tensorflow.keras.layers.Lambda(
    lambda X: tf.compat.v1.reduce_mean(X, axis=[3], keepdims=True))


def depth_pool_selection(i):
    if i == 0:
        return depth_pool_max
    else:
        return depth_pool_mean

###################################################Build Model###################################################

recall, precision, auc = tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.AUC()
def get_model(layer_sizes, kernel_size, kernel_initializer, depth_pool, dropout, learning_rate, activation):
    model = tf.keras.Sequential()
    i = 1
    model.add(tensorflow.keras.layers.Conv2D(layer_sizes[0], kernel_size=kernel_size, activation=activation,
                                             input_shape=sample_shape, kernel_initializer=kernel_initializer,
                                             padding="SAME"))
    #model.add(Depth_pool_selection(reduce_function=depth_pool[0]))
    model.add(depth_pool_selection(depth_pool[0]))
    for layer in layer_sizes[1:]:
        model.add(tensorflow.keras.layers.Conv2D(layer, kernel_size=kernel_size, activation=activation,
                                                 kernel_initializer=kernel_initializer, padding="SAME"))
        #model.add(Depth_pool_selection(reduce_function=depth_pool[i]))
        model.add(depth_pool_selection(depth_pool[0]))
        i = ++i
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tensorflow.keras.layers.Flatten())
    model.add(tensorflow.keras.layers.Dense(256, activation=activation, kernel_initializer='he_uniform'))
    model.add(tensorflow.keras.layers.Dense(no_classes, activation='softmax'))
    # opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    metrics = ['accuracy']
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
    model.summary()
    print("{0}, {1}, {2}, {3}, {4}, {5}, {6}".format(layer_sizes, kernel_size, kernel_initializer, depth_pool, dropout,
                                                     learning_rate, activation))
    return model

###############################################scikit-learn##########################################################
def sklearnGridSearch():
    layer_sizes = [(128, 128, 128), (128, 128, 64, 64), (128, 128, 64, 32), (16, 16, 16, 16), (64, 32, 16)]
    depth_pool = [(0, 0, 0, 0), (1, 1, 1, 1), (0, 1, 0, 1)]
    classifier = tf.keras.wrappers.scikit_learn.KerasClassifier(get_model, batch_size=10)
    validator = sklearn.model_selection.GridSearchCV(classifier,
                                                     param_grid={'epochs': [2],  # , 20, 30],
                                                                 'kernel_size': [3],
                                                                 'kernel_initializer': ['he_uniform'],
                                                                 'depth_pool': depth_pool,
                                                                 'layer_sizes': layer_sizes,
                                                                 'dropout': [x * 0.1 for x in range(0, 2)],
                                                                 'learning_rate': [0.001, 0.01],
                                                                 'activation': ['relu']}, n_jobs=1)

    #log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #validator.fit(x_train, targets_train, validation_data=(x_test, targets_test), callbacks=[tensorboard_callback])
    validator.fit(x_train, y_train)
    print(validator.best_params_)
    print(validator.best_score_)

################################################# H_Param ##############################################################
from tensorboard import program
from _datetime import datetime
from tensorboard.plugins.hparams import api as hp

tb = program.TensorBoard()
#tb.configure(argv=[None, '--logdir', 'C:/Users/Anna/Desktop/Masterarbeit/logs'])
tb.configure(argv=[None, '--logdir', '/home/sc.uni-leipzig.de/ay312doty/logs'])
url = tb.launch()

def split_data(features, labels, test_size):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = test_size, random_state=45)
    return x_train, x_test, y_train, y_test

def train_model_k_fold(x_train, x_test, y_train, y_test, params, log_dir, id):
    units_per_layer = eval(params['units_per_layer']) #conversion back into tuple
    model = get_model(units_per_layer, params['kernel_size'], params['kernel_initializer'],
                      params['depth_pool'], params['dropout'], params['learning_rate'], params['activation'])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir)
    hp_callback = hp.KerasCallback(log_dir, params, trial_id = id)
    #stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 4)
    model.fit(x=x_train,
          y=y_train,
          epochs=params['epochs'],
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback, hp_callback],
          batch_size=32)
    model.save(log_dir + ".h5")
    tf.keras.backend.clear_session()

def train_model_full(x_train, y_train, epochs, params, log_dir, id):
    units_per_layer = eval(params['units_per_layer']) #conversion back into tuple
    model = get_model(units_per_layer, params['kernel_size'], params['kernel_initializer'],
                      params['depth_pool'], params['dropout'], params['learning_rate'], params['activation'])
    model.fit(x=x_train,
          y=y_train,
          epochs=epochs,
          callbacks=[
              tf.keras.callbacks.TensorBoard(log_dir),  # log metrics
              hp.KerasCallback(log_dir, params),  # log hparams
          ],
          batch_size=32)
    model.save(log_dir + ".h5")
    tf.keras.backend.clear_session()

def run(features, labels, parameter_grid):
    log_dir = "logs/hparam1/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    session = 0
    for params in parameter_grid:
        run_name = "run-{0}".format(session)
        train_model_k_fold(x_train, x_test, targets_train, targets_test, params, log_dir + run_name, id = run_name)
        session += 1

layer_sizes = [(128, 128, 64, 64), (16, 16, 16, 16)]
depth_pool = [(0, 0, 0, 0), (1, 1, 1, 1), (0, 1, 0, 1)]
layers_size = list(map(lambda a: str(a), layer_sizes))
depth_pool = list(map(lambda a: str(a), depth_pool))
param_dict = {
    'n_splits': [3],
    'test_size': [0.2],
    'epochs': [20, 30],#[10, 20, 30],
    'kernel_size': [3],# 3],
    'depth_pool': depth_pool,
    'kernel_initializer': ['he_uniform'],
    'units_per_layer': layers_size,
    'dropout': [0],
    'learning_rate': [0.001, 0.01],
    'activation': ['sigmoid','relu']
            }

grid = sklearn.model_selection.ParameterGrid(param_dict)
run(x_train, targets_train, grid)