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

# TODO: Make a run with different parameters and plot results
parameters = {
    'input_layer': None,
    'lstm_layers': [16],
    'n_peers': 2,
    # x, y for 3 targets
    # TODO: Add enabled flag
    'n_input': 3*4,
    # x, y for 1 target. TODO: Add enabled flag.
    'n_output': 2,
    'lstm_clip': 10.0
}

print str(parameters)

generative_model = model.create_generative(parameters)

saver = tf.train.Saver()

# TODO: Work in progress...

# Arbitrary starting positions.
delta = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
pos = np.asarray([[0.0, 0.0], [0.0, 0.3], [0.0, -0.3]])

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
    for time in range(3000):
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
            traces[mod].append(prediction + posForTarget[0,:])
            # The predictions are deltas.
            next_pos.append(prediction + posForTarget[0,:])
            next_delta.append(prediction)
        pos = np.asarray(next_pos)
        delta = np.asarray(next_delta)
        export_to_octave.save('traces.mat', 'traces', np.asarray(traces))
        print 'time: ' + str(time)
