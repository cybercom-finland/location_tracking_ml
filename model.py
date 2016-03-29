#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import pylab

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

import random
import json
import itertools

# Returns the LSTM stack created based on the parameters.
# Processes several batches at once.
# Input shape is: (parameters['batch_size'], parameters['n_steps'], parameters['n_input'])
def RNN(parameters, input, model, initial_state):
    # The model is:
    # 1. input
    # 2. linear layer
    # 3 - n. LSTM layers
    # n+1. linear layer
    # n+1. output
    
    # input shape: (batch_size, n_steps, n_input)
    input = tf.transpose(input, [1, 0, 2])  # permute n_steps and batch_size
    
    # Reshape to prepare input to the linear layer
    input = tf.reshape(input, [-1, parameters['n_input']]) # (n_steps*batch_size, n_input)
    
    # 1. layer, linear activation for each batch and step.
    input = tf.matmul(input, model['input_weights']) + model['input_bias']

    # Split data because rnn cell needs a list of inputs for the RNN inner loop,
    # that is, a n_steps length list of tensors shaped: (batch_size, n_inputs)
    # This is not well documented, but check for yourself here: https://goo.gl/NzA5pX
    input = tf.split(0, parameters['n_steps'], input) # n_steps * (batch_size, :)

    print str(model['rnn_cell'])
    # Note: States is shaped: batch_size x cell.state_size
    outputs, states = rnn.rnn(model['rnn_cell'], input, initial_state=initial_state)
    # Only the last output is interesting for error back propagation and prediction.
    return (tf.matmul(outputs[-1], model['output_weights']) + model['output_bias'], states)

def create(parameters):
    print('Creating the neural network model.')
    
    # tf Graph input
    x = tf.placeholder("float", shape=(None, parameters['n_steps'], parameters['n_input']), name='input')
    y = tf.placeholder("float", shape=(None, parameters['n_output']), name='expected_output')
    lstm_state_size = np.sum(parameters['lstm_layers']) * 2
    # Note: Batch size is the first dimension in istate.
    istate = tf.placeholder("float", shape=(None, lstm_state_size), name='internal_state')
    lr = tf.Variable(parameters['learning_rate'], trainable=False, name='learning_rate')

    # The target to track itself and its peers, each with x, y and velocity x and y.
    input_size = (parameters['n_targets']) * 4
    model = {
        'input_weights': tf.Variable(tf.random_normal([input_size, parameters['input_layer']], stddev=40.0), name='input_weights'),
        'input_bias': tf.Variable(tf.random_normal([parameters['input_layer']], stddev=40.0), name='input_bias'),
        'output_weights': tf.Variable(tf.random_normal([parameters['output_layer'], parameters['n_output']], stddev=40.0),
                                      name='output_weights'),
        'output_bias': tf.Variable(tf.random_normal([parameters['n_output']], stddev=40.0), name='output_bias'),
        'rnn_cell': rnn_cell.MultiRNNCell(
                map(lambda l: rnn_cell.LSTMCell(l, 12, cell_clip=10.0, use_peepholes=True), parameters['lstm_layers'])
            ),
        'lr': lr,
        'x': x,
        'y': y,
        'istate': istate
    }
    
    pred, last_state = RNN(parameters, x, model, istate)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.l2_loss(pred-y)) # L2 loss for regression
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost) # Adam Optimizer
    
    # Evaluate model. This is the average error.
    # We will take 1 m as the arbitrary goal post to be happy with the error.
    error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(pred-y, 2), 1)), 0)
    model['pred'] = pred
    model['last_state'] = last_state
    model['cost'] = cost
    model['optimizer'] = optimizer
    model['error'] = error
    
    return model
