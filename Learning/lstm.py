import sklearn
import tensorflow as tf
import numpy as np
import tensorflow.python.keras.utils.np_utils
import helperClass as help
from tensorflow.keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
from _datetime import datetime
from tensorboard.plugins.hparams import api as hp
from tensorboard import program

def get_model(layer_sizes_conv, layer_sizes_rnn, dropout, epochs, momentum, activation, dense_act):
    model = tf.keras.Sequential()
    model.add(TimeDistributed(tf.keras.layers.Conv2D(filters=layer_sizes_conv[0], kernel_size=(3), padding='same', activation=activation,
                               input_shape=(3, 3, 1))))
    for layer in layer_sizes_conv[1:]:
        model.add(TimeDistributed(tf.keras.layers.Conv2D(filters=layer, kernel_size=(3), padding='same', activation=activation)))
        if dropout:
            model.add(TimeDistributed(tf.keras.layers.Dropout(dropout)))
    model.add(TimeDistributed(tf.keras.layers.BatchNormalization()))
    model.add(TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.SimpleRNN(layer_sizes_rnn[0], unroll=True, return_sequences=True, return_state=False))
    for layer in layer_sizes_rnn[1:]:
        model.add(tf.keras.layers.SimpleRNN(layer, unroll=True))
    model.add(tf.keras.layers.Dense(100, activation=dense_act))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    learning_rate = 0.01
    decay_rate = learning_rate / epochs
    momentum = momentum
    opt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

tb = program.TensorBoard()
#tb.configure(argv=[None, '--logdir', 'C:/Users/Anna/Desktop/Masterarbeit/logs'])
tb.configure(argv=[None, '--logdir', '/home/sc.uni-leipzig.de/ay312doty/logs'])
url = tb.launch()

def train_model_k_fold(x_train, x_test, y_train, y_test, params, log_dir, id):
    model = get_model(eval(params['layer_sizes_conv']), eval(params['layer_sizes_rnn']), params['dropout'], params['epochs'],
                      params['momentum'], params['activation'], params['dense_act'])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir)
    hp_callback = hp.KerasCallback(log_dir, params, trial_id=id)
    # stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 4)
    model.fit(x=x_train,
              y=y_train,
              epochs=params['epochs'],
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback, hp_callback],
              batch_size=params['batch_size'])
    model.save(log_dir + ".h5")
    tf.keras.backend.clear_session()

x, y = help.load3Ddata()
x, y = help.balanced_dataset(x, y, [0, 2])
x = x[y < 3]
y = y[y < 3]
sample_shape = (61, 3, 3)
x = x.reshape(x.shape[0], 61, 3, 3, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

targets_train = tensorflow.keras.utils.to_categorical(y_train).astype(np.integer)
targets_test = tensorflow.keras.utils.to_categorical(y_test).astype(np.integer)

def run(parameter_grid):
    log_dir = "logs/hparam_lstm/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    session = 0
    for params in parameter_grid:
        run_name = "run-{0}".format(session)
        train_model_k_fold(x_train, x_test, targets_train, targets_test, params, log_dir + run_name, id=run_name)
        session += 1

layer_sizes_conv = [(32, 16), (64), (64, 32)]
layer_sizes_rnn = [(512, 256), (256, 128)]
layer_sizes_conv = list(map(lambda a: str(a), layer_sizes_conv))
layer_sizes_rnn = list(map(lambda a: str(a), layer_sizes_rnn))

param_dict = {
    'batch_size': [100, 150],
    'epochs': [30],
    'layer_sizes_conv': layer_sizes_conv,
    'layer_sizes_rnn': layer_sizes_rnn,
    'momentum': [0.8],
    'dropout': [0, 0.1],
    'activation': ['relu', 'tanh'],
    'dense_act': ['relu', 'tanh']
}

grid = sklearn.model_selection.ParameterGrid(param_dict)
run(grid)
