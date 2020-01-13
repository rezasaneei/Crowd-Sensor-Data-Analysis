# -*- coding: utf-8 -*-
"""
This code is a modified and improved version of the work of Magnus Erik Hvass
Pedersen that is freely available at https://github.com/Hvass-Labs/ under MIT 
license.Completely authorized for publication in its original form or its 
modified versions.
The copyright is provided below:
License (MIT)
Copyright (c) 2018 by Magnus Erik Hvass Pedersen
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

The code is modified with the new hyperparameter optimization capabilities
that let the user tune the model with the HOP algorithms such as RandomSearch
BayesianOptimization, etc. We have used BOHB for this purpose. This code is 
, in fact, the objective function of our BOHB algorithm.

Concretely we enhanced the code with:
Hyper parameter adjustment through a dictionary
Architecture selection including:
    Neural models: LSTM, GRU
    Optional Number of layers, Number of units, learning rate, and optimizer.
Smoothing function that returns a smoothed loss to deal with data noise.
Modularity: To let the code accept external parameters, and be wrapped in under 
an optimizer algorithm.

This code has been later modified by:
Reza Saneei
"""


"""
Index(['Clear', 'Clouds', 'Drizzle', 'Fog', 'Mist', 'Rain', 'Snow',
       'Thunderstorm', 'temp', 'wind_speed', 'rain_1h', 'clouds_all',
       'GF-Ferjeleia', 'GF-Glasshytta', 'GF-Isegran', 'GF-Kasernen',
       'GF-Laboratoriegaten', 'GF-Magenta', 'GF-Mormors', 'GF-Pumpehuset',
       'Totalt', 'year_day', 'week_day', 'hour'],
      dtype='object')

"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
#import os
from sklearn.preprocessing import MinMaxScaler

# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, LSTM, Dropout
from tensorflow.python.keras.optimizers import RMSprop, Adam, Adagrad, SGD
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, LambdaCallback, Callback


# Hyperparameters
hparams_dict = {
    'epochs': 100,
    'rnn_type': 'GRU',
    'rnn_size': 1024,
    'learning_rate': 1e-3,
    'batch_size': 64,
    'sequence_length': 24 * 7 * 1,
    'steps_per_epoch': 30,
    'layers': 2,
    'dropout': 0.31,
    'warmup_steps': 50,
    'optimizer_type': 'Adagrad',
    'weight_initialization': False}

{"batch_size": 6,
 "dropout": 0.3110996040539636,
 "layers": 2,
 "optimizer_type": "Adagrad",
 "rnn_size": 10,
 "rnn_type": "GRU",
 "sequence_length": 2}

"""
target_names = ['GF-Ferjeleia', 'GF-Glasshytta', 'GF-Isegran', 'GF-Kasernen', 
'GF-Laboratoriegaten', 'GF-Magenta', 'GF-Mormors', 'GF-Pumpehuset', 'Totalt']
"""
target_names = ['Totalt']


## Load Data

df = pd.read_csv('weather-traffic-dummy_reza.csv')
#df['Date'] = pd.to_datetime(df['Date'], format= '%Y-%m-%d %H:%M:%S')
df = df.drop(['DateIndex'], axis=1)
df.set_index('Index', inplace=True)

# we shift the data to let the model to learn to predict (forecast)
shift_days = 1
shift_steps = shift_days * 24  # Number of hours.

df_targets = df[target_names].shift(-shift_steps)

# Input signals
x_data = df.values[0:-shift_steps]
print(type(x_data))
print("Shape:", x_data.shape)

# output signals
y_data = df_targets.values[:-shift_steps]
print(type(y_data))
print("Shape:", y_data.shape)


# specifies the train and test data sizes
train_test_fraction = 0.9
data_size = len(x_data)

train_size = int(train_test_fraction * data_size)
test_size = data_size - train_size

# creates dataframes for train and test sets
x_train = x_data[0:train_size].copy()
x_test = x_data[train_size:].copy()
print(len(x_train),  len(x_test))
print(len(x_train) + len(x_test))

y_train = y_data[0:train_size].copy()
y_test = y_data[train_size:].copy()
print(len(y_train),  len(y_test))
print(len(y_train) + len(y_test))

# the input and output dimensions
x_features = x_data.shape[1]
y_features = y_data.shape[1]


test_train_length = [len(x_test), len(x_train)]

# scaling the input data between 0 and 1
x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)

x_test_scaled = x_scaler.transform(x_test)

# scaling the output data between 0 and 1
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

"""## Data Generator"""


def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, x_features)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, y_features)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(train_size - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]

        yield (x_batch, y_batch)


def mse_loss(y_true, y_pred):
    """
    returns the Mean Squared Loss between the real output and model's prediction
    Args:
        y_true
    """
    warmup_steps = 50

    # skips the first warmup_stemps
    y_true = y_true[:, warmup_steps:, :]
    y_pred = y_pred[:, warmup_steps:, :]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.reduce_mean(tf.losses.mean_squared_error(
        labels=y_true, predictions=y_pred))

    return loss


def smooth_data(data, smooth=4, extension=True):

    # Compute moving averages using different window sizes
    start_extension = [data[0]]*smooth
    end_extension = [data[-1]]*smooth
    # extend the line in the left and right
    if extension:
        data = start_extension + data + end_extension

    N = len(data)
    y = np.asarray(data)
    y_avg = np.zeros((1, N))
    #ax.plot(x, y)

    avg_mask = np.ones(smooth) / smooth
    y_avg = np.convolve(y, avg_mask, 'same')
    if extension:
        data = y_avg[smooth:-smooth]
    else:
        data = y_avg
    return data


def main(hparams_dict):

    # hparams_dict ={ \
    # 'epochs' : 5, \
    # 'rnn_type' : 'LSTM', \
    # 'rnn_size' : 8, \
    # 'learning_rate' : 1e-3, \
    # 'batch_size' : 64, \
    # 'sequence_length' : 24 * 7 * 1, \
    # 'steps_per_epoch' : 2, \
    # 'layers' : 1, \
    # 'dropout' : 0.5, \
    # 'warmup_steps' : 50, \
    # 'optimizer_type' : 'RMSprop', \
    # 'weight_initialization' : False}

    epochs = hparams_dict['epochs']
    rnn_type = hparams_dict['rnn_type']
    rnn_size = hparams_dict['rnn_size']
    learning_rate = hparams_dict['learning_rate']
    batch_size = hparams_dict['batch_size']
    sequence_length = hparams_dict['sequence_length']
    steps_per_epoch = hparams_dict['steps_per_epoch']
    layers = hparams_dict['layers']
    dropout = hparams_dict['dropout']
    warmup_steps = hparams_dict['warmup_steps']
    optimizer_type = hparams_dict['optimizer_type']
    weight_initialization = hparams_dict['weight_initialization']

    model_name = '{}_Multi-{}_epoch-{}'.format(
        rnn_type, len(target_names), str(epochs))

    # creates a generator object from the batch generator
    generator = batch_generator(
        batch_size=batch_size, sequence_length=sequence_length)



    # converts the 1 dimensional vector to 2 dimensional e.g. (2,) >>> (2,1)
    validation_data = (np.expand_dims(x_test_scaled, axis=0),
                       np.expand_dims(y_test_scaled, axis=0))

    # creates the Neural Model with keras
    model = Sequential()

    for layer in range(layers):
        if rnn_type == 'LSTM':
            model.add(LSTM(units=rnn_size,
                           return_sequences=True,
                           input_shape=(None, x_features,)))
        if rnn_type == 'GRU':
            model.add(GRU(units=rnn_size,
                          return_sequences=True,
                          input_shape=(None, x_features,)))

    model.add(Dropout(dropout))

    dense_layer = Dense(y_features, activation='sigmoid')

    model.add(dense_layer)

    #Uses a linear activation in case of using weight_initialization
    if weight_initialization:
        from tensorflow.python.keras.initializers import RandomUniform

        # Maybe use lower init-ranges.
        init = RandomUniform(minval=-0.05, maxval=0.05)

        model.add(Dense(y_features,
                        activation='linear',
                        kernel_initializer=init))

    # Decides on the type of optimizer based on HPs
    if optimizer_type == 'RMSprop':
        optimizer = RMSprop(lr=learning_rate)
    if optimizer_type == 'Adagrad':
        optimizer = Adagrad(lr=learning_rate)
    if optimizer_type == 'SGD':
        optimizer = SGD(lr=learning_rate)

    # compiles the model
    model.compile(loss=mse_loss, optimizer=optimizer)

    print(model.summary())

    # callbacks for save the checkpoint, earlyStopping, the tensorboard log, redusing lr, and a csv file

    path_checkpoint = model_name + '.keras'
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)

    callback_early_stopping = EarlyStopping(monitor='val_loss',
                                            patience=5, verbose=1)

    callback_tensorboard = TensorBoard(log_dir='./' + model_name + '_logs/',
                                       histogram_freq=0,
                                       write_graph=False)

    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           min_lr=1e-4,
                                           patience=0,
                                           verbose=1)

    callback_csv_log = CSVLogger('training_log.log')

    # saves all the metrics in a list
    metrics_history = []
    metrics_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: metrics_history.append(
            [logs['loss'], logs['val_loss']])
    )

    callbacks = [  # callback_early_stopping,
        callback_checkpoint,
        callback_tensorboard,
        callback_reduce_lr,
        # callback_csv_log,
        metrics_callback
    ]

    # trains the compiled model
    model.fit_generator(generator=generator,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=validation_data,
                        callbacks=callbacks)

    # in case of testing pretrained model
    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    # evaluates the model on the test data
    result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                            y=np.expand_dims(y_test_scaled, axis=0))

    # stores the training and validation losses in two separate lists
    train_loss = [value[0] for value in metrics_history]
    val_loss = [value[1] for value in metrics_history]

    metrics_history = {'train_loss': train_loss, 'val_loss': val_loss}
    return metrics_history


if __name__ == '__main__':
    print(main(hparams_dict))
