#!/usr/bin/python

import matplotlib
from boto.gs.acl import SCOPE
matplotlib.use('Agg')
import pylab

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

import random
import json
import itertools

epsilon = 0.00001

# The input shape is [batch_size, n_mixtures * 6]
def splitMix(output, n_mixtures, batch_size):
    out_pi = output[0:batch_size, 0:n_mixtures]
    out_sigma = tf.reshape(output[0:batch_size, n_mixtures: n_mixtures * 3], [batch_size, n_mixtures, 2])
    out_mu = tf.reshape(output[0:batch_size, n_mixtures * 3 : n_mixtures * 5], [batch_size, n_mixtures, 2])
    out_rho = output[0:batch_size, n_mixtures * 5 : n_mixtures * 6]
    return out_pi, out_sigma, out_mu, out_rho

# The result shape is [batch_size, n_mixtures * 6]
def joinMix(out_pi, out_sigma, out_mu, out_rho, n_mixtures, batch_size):
    return tf.concat(1, [out_pi, tf.reshape(out_sigma, [batch_size, n_mixtures * 2]), \
                      tf.reshape(out_mu, [batch_size, n_mixtures * 2]), out_rho])

# Returns the softmaxed mixture coefficients (weight, 2xstandard deviation, 2xmean and correlation)
# The output must be a tensor of batches, so that each batch has
# weight, deviation and mean triplets for one target variable.
# output shape is [batch_size, n_mixtures]
# Note: For one 2D variable only.
def softmax_mixtures(output, n_mixtures, batch_size):
    out_pi, out_sigma, out_mu, out_rho = splitMix(output, n_mixtures, batch_size)

    # Softmaxing the weights so that they sum up to one.
    out_pi = tf.nn.softmax(tf.log(epsilon + tf.abs(out_pi)))

    # Always [-0.2, 0.2]
    out_rho = tf.tanh(out_rho) * 0.2
    #out_rho = tf.zeros([batch_size, n_mixtures]) # Uncorrelated

    # Making sigma always positive and between some sane values (0.018316, 54.598).
    out_sigma = tf.exp(tf.tanh(out_sigma) * 4)

    return joinMix(out_pi, out_sigma, out_mu, out_rho, n_mixtures, batch_size)

# Returns the probability density for bivariate gaussians.
# mu is x,y pairs of mus for each mixture gaussian, and for each batch.
# sigma is x,y pairs of sigmas for each mixture gaussian, and for each batch.
# The first dimension is batch, then mixture, then variable.
# Rho is the correlation of x and y for each batch and mixture.
# The first dimension is batch, then mixture.
def tf_bivariate_normal(y, mu, sigma, rho, n_mixtures, batch_size):
    mu = tf.verify_tensor_all_finite(mu, "Mu not finite!")
    y = tf.verify_tensor_all_finite(y, "Y not finite!")
    delta = tf.sub(tf.tile(tf.expand_dims(y, 1), [1, n_mixtures, 1]), mu)
    delta = tf.verify_tensor_all_finite(delta, "Delta not finite!")
    sigma = tf.verify_tensor_all_finite(sigma, "Sigma not finite!")
    s = tf.abs(tf.reduce_prod(sigma, 2))
    # s >= 0
    s = tf.verify_tensor_all_finite(s, "S not finite!")
    # -1 <= rho <= 1
    z = tf.reduce_sum(tf.square(tf.mul(delta, tf.inv(sigma + epsilon))), 2) - \
        2 * tf.mul(tf.mul(rho, tf.reduce_prod(delta, 2)), tf.inv(s + epsilon))
    z = tf.verify_tensor_all_finite(z, "Z not finite!")
    # 0 < negRho <= 1
    negRho = (1 - tf.square(rho)) # * 0.5 + 0.5
    negRho = tf.verify_tensor_all_finite(negRho, "negRho not finite!")
    #negRho = tf.ones([batch_size, n_mixtures]) # Uncorrelated
    # Note that if negRho goes near zero, or z goes really large, this explodes.
    result = tf.exp(tf.mul(-z, tf.inv(2 * negRho)))
    result = tf.verify_tensor_all_finite(result, "Result in bivariate normal not finite!")
    denom = 2 * np.pi * tf.mul(s, tf.sqrt(negRho))
    denom = tf.verify_tensor_all_finite(denom, "Denom in bivariate normal not finite!")
    result = tf.mul(result, tf.inv(denom + epsilon))
    result = tf.verify_tensor_all_finite(result, "Result2 in bivariate normal not finite!")
    return result

def mixture_loss(pred, y, n_mixtures, batch_size):
    pred = tf.verify_tensor_all_finite(pred, "Pred not finite!")
    out_pi, out_sigma, out_mu, out_rho = splitMix(pred, n_mixtures, batch_size)
    tf.Assert(tf.greater(out_pi, 0), [out_pi])
    tf.Assert(tf.greater(out_sigma, 0), [out_sigma])
    tf.Assert(tf.less(out_rho, 1), [out_rho])
    tf.Assert(tf.greater(out_rho, -1), [out_rho])
    result_binorm = tf_bivariate_normal(y, out_mu, out_sigma, out_rho, n_mixtures, batch_size)
    result_binorm = tf.Print(result_binorm, [tf.reduce_sum(result_binorm)], "Result: ")
    
    result_binorm = tf.verify_tensor_all_finite(result_binorm, "Result not finite1!")
    result_weighted = tf.mul(result_binorm, out_pi)
    result_weighted = tf.verify_tensor_all_finite(result_weighted, "Result not finite2!")
    result_raw = tf.reduce_sum(result_weighted, 1, keep_dims=True)
    result_raw = tf.Print(result_raw, [result_raw], "Result2: ")
    result_raw = tf.verify_tensor_all_finite(result_raw, "Result not finite3!")
    result = -tf.log(result_raw + epsilon)
    tf.Assert(tf.greater(result, 0), [result])
    result = tf.Print(result, [result], "Result3: ")
    result = tf.verify_tensor_all_finite(result, "Result not finite4!")
    # Adding additional error terms to prevent numerical instability for flat gradients for sigma and rho.
    #s = tf.abs(tf.reduce_prod(out_sigma, 2))
    #result = result + (tf.square(out_rho) + tf.inv(s + epsilon) + tf.inv(out_pi + epsilon)) * 0.000001
    result = tf.reduce_sum(result)
    result = tf.Print(result, [result], "Result4: ")
    result = tf.verify_tensor_all_finite(result, "Result not finite5!")
    return result

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
    input = tf.verify_tensor_all_finite(input, "Input not finite!")
    # input shape: (batch_size, n_steps, n_input)
    input = tf.transpose(input, [1, 0, 2])  # permute n_steps and batch_size
    input = tf.verify_tensor_all_finite(input, "Input not finite2!")
    
    # Reshape to prepare input to the linear layer
    input = tf.reshape(input, [-1, parameters['n_input']]) # (n_steps*batch_size, n_input)
    input = tf.verify_tensor_all_finite(input, "Input not finite3!")
    
    # 1. layer, linear activation for each batch and step.
    if (model.has_key('input_weights')):
        model['input_weights'] = tf.clip_by_value(model['input_weights'], -parameters['lstm_clip'], parameters['lstm_clip'])
        model['input_bias'] = tf.clip_by_value(model['input_bias'], -parameters['lstm_clip'], parameters['lstm_clip'])
        input = tf.matmul(input, model['input_weights']) + model['input_bias']

    # Split data because rnn cell needs a list of inputs for the RNN inner loop,
    # that is, a n_steps length list of tensors shaped: (batch_size, n_inputs)
    # This is not well documented, but check for yourself here: https://goo.gl/NzA5pX
    input = tf.split(0, parameters['n_steps'], input) # n_steps * (batch_size, :)

    initial_state = tf.verify_tensor_all_finite(initial_state, "Initial state not finite!")
    # Note: States is shaped: batch_size x cell.state_size
    outputs, states = rnn.rnn(model['rnn_cell'], input, initial_state=initial_state)
    outputs[-1] = tf.verify_tensor_all_finite(outputs[-1], "Outputs not finite!")
    # Only the last output is interesting for error back propagation and prediction.
    # Note that all batches are handled together here.

    model['output_weights'] = tf.clip_by_value(model['output_weights'], -parameters['lstm_clip'], parameters['lstm_clip'])
    model['output_bias'] = tf.clip_by_value(model['output_bias'], -parameters['lstm_clip'], parameters['lstm_clip'])
    model['output_weights'] = tf.verify_tensor_all_finite(model['output_weights'], "Output weights not finite!")
    model['output_bias'] = tf.verify_tensor_all_finite(model['output_bias'], "Output biases not finite!")
    raw_output = tf.matmul(outputs[-1], model['output_weights']) + model['output_bias']
    raw_output = tf.verify_tensor_all_finite(raw_output, "Raw output not finite!")
    
    n_mixtures = parameters['n_mixtures']
    batch_size = parameters['batch_size']
    # And now, instead of just outputting the expected value, we output mixture distributions.
    # The number of mixtures is intuitively the number of possible actions the target can take.
    # The output is divided into triplets of n_mixtures mixture parameters for the 2 absolute position coordinates.
    output = softmax_mixtures(raw_output, n_mixtures, batch_size)
    output = tf.verify_tensor_all_finite(output, "Final output not finite!")

    return (output, states)

# Returns the generative LSTM stack created based on the parameters.
# Processes one input at a time.
# Input shape is: 1 x (parameters['n_input'])
# State shape is: 1 x (parameters['n_input'])
def RNN_generative(parameters, input, model, initial_state):
    # The model is:
    # 1. input
    # 2. linear layer
    # 3 - n. LSTM layers
    # n+1. linear layer
    # n+1. output
    
    # input shape: (1 x n_input)
    
    # 1. layer, linear activation for each batch and step.
    if (model.has_key('input_weights')):
        input = tf.matmul(input, model['input_weights']) + model['input_bias']

    # Note: States is shaped: batch_size x cell.state_size
    # Input should be a tensor of [batch_size, depth]
    # State should be a tensor of [batch_size, depth]
    outputs, states = model['rnn_cell'](input, initial_state)
    raw_output = tf.matmul(outputs, model['output_weights']) + model['output_bias']
    # TODO: Sample distribution here: First select the mixture based on weights, then sample normal.
    # Tanh for the delta components
    #output = tf.concat(1, [raw_output[:,0:2], tf.tanh(raw_output[:,2:4])])
    return (output, states)

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
    input_size = (parameters['n_peers'] + 1) * 4
    inputToRnn = parameters['input_layer']
    if (parameters['input_layer'] == None):
        inputToRnn = parameters['n_input']

    cells = [rnn_cell.LSTMCell(l, parameters['lstm_layers'][i-1] if (i > 0) else inputToRnn,
                               cell_clip=parameters['lstm_clip'], use_peepholes=True) for i,l in enumerate(parameters['lstm_layers'])] 
    # TODO: GRUCell cupport here.
    # cells = [rnn_cell.GRUCell(l, parameters['lstm_layers'][i-1] if (i > 0) else inputToRnn) for i,l in enumerate(parameters['lstm_layers'])]
    model = {
        'output_weights': tf.Variable(tf.random_normal([parameters['lstm_layers'][-1],
                                                        # 6 = 2 sigma, 2 mean, weight, rho
                                                        parameters['n_mixtures'] * 6]),
                                      name='output_weights'),
        # We need to put at least the standard deviation output biases to about 5 to prevent zeros and infinities.
        # , mean = 5.0, stddev = 3.0
        'output_bias': tf.Variable(tf.random_normal([parameters['n_mixtures'] * 6]),
                                   name='output_bias'),
        'rnn_cell': rnn_cell.MultiRNNCell(cells),
        'lr': lr,
        'x': x,
        'y': y,
        'istate': istate
    }
    if (parameters['input_layer'] <> None):
        model['input_weights'] = tf.Variable(tf.random_normal(
            [input_size, parameters['input_layer']]), name='input_weights')
        model['input_bias'] = tf.Variable(tf.random_normal([parameters['input_layer']]), name='input_bias')

    
    pred = RNN(parameters, x, model, istate)
    
    for variable in tf.trainable_variables():
        variable = tf.verify_tensor_all_finite(variable, "Trainable variable not finite!")
        
    # Define loss and optimizer
    # We will take 1 m as the arbitrary goal post to be happy with the error.
    # The delta error is taken in squared to emphasize its importance (errors are much smaller than in absolute
    # positions)
    n_mixtures = parameters['n_mixtures']
    batch_size = parameters['batch_size']
    
    # RNN/MultiRNNCell/Cell1/LSTMCell/W_O_diag
    # RNN/MultiRNNCell/Cell1/LSTMCell/W_I_diag
    # RNN/MultiRNNCell/Cell1/LSTMCell/W_F_diag
    # RNN/MultiRNNCell/Cell1/LSTMCell/W_0
    # RNN/MultiRNNCell/Cell1/LSTMCell/B
        
    cost = mixture_loss(pred[0], y, n_mixtures, batch_size)

    # The gradients go fast to infinity in certain points if not clipped.
    #grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 1.0)
    # tf.train.AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost) # Adam Optimizer
    # train_op = optimizer.apply_gradients(zip(grads, tvars))
     
    model['pred'] = pred[0]
    model['last_state'] = pred[1]
    model['cost'] = cost
    model['optimizer'] = optimizer
    
    return model

def create_generative(parameters):
    print('Creating the generative neural network model.')
    
    # tf Graph input
    # TODO: Technically we could run all the modules in the bank inside one batch here.
    x = tf.placeholder("float", shape=(1, parameters['n_input']), name='input')
    y = tf.placeholder("float", shape=(1, parameters['n_output']), name='expected_output')
    lstm_state_size = np.sum(parameters['lstm_layers']) * 2
    # Note: Batch size is the first dimension in istate.
    istate = tf.placeholder("float", shape=(None, lstm_state_size), name='internal_state')

    # The target to track itself and its peers, each with x, y and velocity x and y.
    input_size = (parameters['n_peers'] + 1) * 4
    inputToRnn = parameters['input_layer']
    if (parameters['input_layer'] == None):
        inputToRnn = parameters['n_input']

    cells = [rnn_cell.LSTMCell(l, parameters['lstm_layers'][i-1] if (i > 0) else inputToRnn,
                               cell_clip=parameters['lstm_clip'], use_peepholes=True) for i,l in enumerate(parameters['lstm_layers'])] 
    # TODO: GRUCell support here.
    # cells = [rnn_cell.GRUCell(l, parameters['lstm_layers'][i-1] if (i > 0) else inputToRnn) for i,l in enumerate(parameters['lstm_layers'])]
    model = {
        'output_weights': tf.Variable(tf.random_normal([parameters['lstm_layers'][-1],
                                                        parameters['n_output'] * parameters['n_mixtures']]),
                                      name='output_weights'),
        'output_bias': tf.Variable(tf.random_normal([parameters['n_output'] * parameters['n_mixtures']]), name='output_bias'),
        'rnn_cell': rnn_cell.MultiRNNCell(cells),
        'x': x,
        'y': y,
        'istate': istate
    }
    
    if (parameters['input_layer'] <> None):
        model['input_weights'] = tf.Variable(tf.random_normal(
            [input_size, parameters['input_layer']]), name='input_weights')
        model['input_bias'] = tf.Variable(tf.random_normal([parameters['input_layer']]), name='input_bias')

    # The next variables need to be remapped, because we don't have RNN context anymore:
    # RNN/MultiRNNCell/Cell0/LSTMCell/ -> MultiRNNCell/Cell0/LSTMCell/
    # B, W_F_diag, W_O_diag, W_I_diag, W_0
    with tf.variable_scope("RNN"):
        pred = RNN_generative(parameters, x, model, istate)
    
    # Define loss and optimizer
    # cost = tf.reduce_mean(tf.sqrt(tf.nn.l2_loss(pred-y)) # L2 loss for regression
    # Evaluate model. This is the average error.
    # We will take 1 m as the arbitrary goal post to be happy with the error.
    error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(pred[0]-y, 2), 1)), 0)
    
    model['pred'] = pred
    model['last_state'] = pred[1]
    model['error'] = error

    return model
