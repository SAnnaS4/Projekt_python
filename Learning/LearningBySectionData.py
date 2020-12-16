import sklearn.model_selection
import tensorflow as tf
import numpy as np
import tensorflow.python.keras
import tensorflow.python.keras.utils.np_utils

#import tensorflow.python.layers.base
from sklearn.model_selection import train_test_split
import helperClass as help

config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  )
config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

x, y = help.load3Ddata()

# 1 => EAC
# 2 => Stroma
# 3 => Plattenepithel
# 4 => Blank

x, y = help.balanced_dataset(x, y, [0, 2])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# -- Preparatory code --
# Model configuration
batch_size = 10
nr_epochs = 10
#learning_rate = 0.002
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
def get_model(layer_sizes, kernel_initializer, depth_pool, activation, opt):
    model = tf.keras.Sequential()
    i = 1
    model.add(tensorflow.keras.layers.Conv2D(layer_sizes[0], kernel_size=3, activation=activation,
                                             input_shape=sample_shape, kernel_initializer=kernel_initializer,
                                             padding="SAME"))
    #model.add(Depth_pool_selection(reduce_function=depth_pool[0]))
    model.add(depth_pool_selection(depth_pool[0]))
    for layer in layer_sizes[1:]:
        model.add(tensorflow.keras.layers.Conv2D(layer, kernel_size=3, activation=activation,
                                                 kernel_initializer=kernel_initializer, padding="SAME"))
        #model.add(Depth_pool_selection(reduce_function=depth_pool[i]))
        model.add(depth_pool_selection(depth_pool[0]))
        i = ++i
    model.add(tensorflow.keras.layers.Flatten())
    model.add(tensorflow.keras.layers.Dense(256, activation=activation, kernel_initializer=kernel_initializer))
    model.add(tensorflow.keras.layers.Dense(no_classes, activation='softmax'))
    metrics = ['accuracy']
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=metrics)
    model.summary()
    print("{0}, {1}, {2}, {3}".format(layer_sizes, kernel_initializer, depth_pool, activation))
    return model

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
    #declare Opt
    momentum = 0.8
    learning_rate = 0.1
    decay_rate = learning_rate / 30
    sgd = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    rmsProp = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    if(params['optimizer'] == 1):
        opt = sgd
    elif(params['optimizer'] == 2):
        opt = rmsProp
    else: opt = adam
    units_per_layer = eval(params['units_per_layer']) #conversion back into tuple
    model = get_model(units_per_layer, params['kernel_initializer'], params['depth_pool'], params['activation'],
                      opt)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir)
    hp_callback = hp.KerasCallback(log_dir, params, trial_id = id)
    model.fit(x=x_train,
          y=y_train,
          epochs=params['epochs'],
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback, hp_callback],
          batch_size=params['batch_size'])
    model.save(log_dir + ".h5")
    tf.keras.backend.clear_session()

def run(parameter_grid):
    log_dir = "logs/hparam1/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    session = 0
    for params in parameter_grid:
        run_name = "run-{0}".format(session)
        train_model_k_fold(x_train, x_test, targets_train, targets_test, params, log_dir + run_name, id = run_name)
        session += 1

layer_sizes = [(128, 128, 64, 64)]
depth_pool = [(1, 1, 1, 1)]
layers_size = list(map(lambda a: str(a), layer_sizes))
depth_pool = list(map(lambda a: str(a), depth_pool))


param_dict = {
    'batch_size': [16],
    'epochs': [30],
    'depth_pool': depth_pool,
    'kernel_initializer': ['he_uniform'],
    'units_per_layer': layers_size,
    'learning_rate': [0.001],
    'activation': ['selu'],
    'optimizer': [3]
            }

grid = sklearn.model_selection.ParameterGrid(param_dict)
run(grid)