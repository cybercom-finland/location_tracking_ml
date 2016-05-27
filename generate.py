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

def get_pi_idx(x, pdf):
    N = 2
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
            return i
    print 'error with sampling ensemble'
    return -1

def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
    mean = [mu1, mu2]
    cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


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
            #deltaForTarget = np.copy(delta)
            # Switching the target first.
            posForTarget[[mod, 0], :] = posForTarget[[0, mod], :]
            input = np.asarray(posForTarget)[0:2, :]
            (pred, bank[mod]) = sess.run([generative_model['pred'], generative_model['last_state']], feed_dict={
                                             generative_model['x']: np.reshape(input, (1, 4)),
                                             generative_model['istate']: bank[mod]})
            pred = np.asarray(pred)
            idx = get_pi_idx(random.random(), pred[0,0:2])
            prediction = sample_gaussian_2d(pred[0,6 + idx], pred[0,8 + idx], pred[0,2 + idx], pred[0,4 + idx],
                                       pred[0,10 + idx])
            traces[mod].append(prediction)
            # The predictions are positions and deltas.
            # They will have a difference, which will be used as random noise to the absolute location.
            # The standard deviation of the noise will be at least 0.05.
            # As the network estimates both the delta and the absolute position, we can see the difference
            # and use that metric of uncertainty as a basis for the noise.
            #error_xv = np.random.normal(0, sigma_x)
            #error_yv = np.random.normal(0, sigma_y)
            next_pos.append(prediction)
            #next_delta.append(prediction[0,2:4] + [error_xv, error_yv])
        pos = np.asarray(next_pos)
        delta = np.asarray(next_delta)
        export_to_octave.save('traces.mat', 'traces', np.asarray(traces))
        print 'time: ' + str(time)
