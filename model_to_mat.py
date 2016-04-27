#!/usr/bin/python

import traceback

import matplotlib
matplotlib.use('Agg')
import pylab

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

import random
import json
import itertools

import manage_data
import export_to_octave
import model

import params

parameters = params.parameters

network_model = model.create(parameters)

init = tf.initialize_all_variables()
saver = tf.train.Saver(tf.all_variables())
# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, 'soccer-model')
    print "Saving vars."
    avars = tf.all_variables()
    for variable in avars:
        export_to_octave.save(variable.name.replace('/', '_') + '.mat', 'd', np.asarray(variable.eval()))
    print "All saved."
