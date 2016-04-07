#!/usr/bin/python -u

import matplotlib
matplotlib.use('Agg')
import pylab

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

import random
import json
import itertools

import load_data
import export_to_octave
import manage_data
import model
import train

import params

parameters = params.parameters

print str(parameters)

generative_model = model.create_generative(parameters)

saver = tf.train.Saver()

# TODO: Work in progress...

# Arbitrary starting positions.
delta = np.asarray([[0.15, 0.0], [0.0, -0.1], [-0.1, 0.1]])
pos = np.asarray([[0.0, 0.0], [0.1, 1.3], [10.1, -1.3]])

# For outputting to plot and Octave
traces = [[],[],[]]

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "soccer-model")
    print("Model restored.")
    # Internal states for the three LSTM modules.
    bank = [np.asarray(generative_model['rnn_cell'].zero_state(1, tf.float32).eval()),
            np.asarray(generative_model['rnn_cell'].zero_state(1, tf.float32).eval()),
            np.asarray(generative_model['rnn_cell'].zero_state(1, tf.float32).eval())]
    
    # 5 minutes
    for time in range(600*5):
        next_pos = []
        next_delta = []
        for mod in range(3):
            posForTarget = np.copy(pos)
            deltaForTarget = np.copy(delta)
            # Switching the target first.
            posForTarget[[mod, 0], :] = posForTarget[[0, mod], :]
            input = np.asarray([np.concatenate((posForTarget, deltaForTarget), axis=1)])
            (prediction, bank[mod]) = sess.run(generative_model['pred'], feed_dict={
                                             generative_model['x']: np.reshape(input, (1, 12)),
                                             generative_model['istate']: bank[mod]})
            traces[mod].append(prediction)
            # The predictions are positions and deltas.
            # They will have a difference, which will be used as random noise to the absolute location.
            # The standard deviation of the noise will be at least 0.05.
            # As the network estimates both the delta and the absolute position, we can see the difference
            # and use that metric of uncertainty as a basis for the noise.
            sigma_x = 0.001
            sigma_y = 0.001
            error_x = np.random.normal(0, sigma_x)
            error_y = np.random.normal(0, sigma_y)
            error_xv = np.random.normal(0, sigma_x)
            error_yv = np.random.normal(0, sigma_y)
            next_pos.append(prediction[0,0:2] + [error_x, error_y])
            next_delta.append(prediction[0,2:4] + [error_xv, error_yv])
        pos = np.asarray(next_pos)
        delta = np.asarray(next_delta)
        export_to_octave.save('traces.mat', 'traces', np.asarray(traces))
        print 'time: ' + str(time)
