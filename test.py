#!/usr/bin/python -u

# This file has some unit tests.

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

import load_data
import export_to_octave
import manage_data
import model
import train

import params

batch_size = 1000
n_mixtures = 2

x = tf.placeholder(tf.float32, shape=(batch_size, n_mixtures * 6), name='x')
out_pi, out_sigma, out_mu, out_rho = model.splitMix(x, n_mixtures, batch_size)
y = model.joinMix(out_pi, out_sigma, out_mu, out_rho, n_mixtures, batch_size)

small_x = tf.placeholder(tf.float32, shape=(2, 2 * 6), name='small_x')
small_pi, small_sigma, small_mu, small_rho = model.splitMix(small_x, 2, 2)

actual = tf.placeholder(tf.float32, shape=(batch_size, 2), name='actual')
mu = tf.placeholder(tf.float32, shape=(batch_size, n_mixtures, 2), name='mu')
sigma = tf.placeholder(tf.float32, shape=(batch_size, n_mixtures, 2), name='sigma')
rho = tf.placeholder(tf.float32, shape=(batch_size, n_mixtures), name='rho')
normalValues = model.tf_bivariate_normal(actual, mu, sigma, rho, n_mixtures, batch_size)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    # Asserting that splitMix and joinMix work.
    inValue = np.random.rand(batch_size, n_mixtures * 6)
    outValue = sess.run(y, feed_dict = {x: inValue})
    differenceFromEqual = np.max(np.abs(outValue - inValue))
    if (differenceFromEqual > 1e-7):
        print "FAILED! In value did not equal out value! Difference: " + str(differenceFromEqual)
    else:
        print "OK"
    
    inValue = [
               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
               [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]
               ]
    p, s, m, r = sess.run([small_pi, small_sigma, small_mu, small_rho], feed_dict = {small_x: inValue})
    expectedP = [[  1.0,   2.0],
                 [ 21.0,  22.0]]
    expectedS = [[[  3.0,   4.0],
                  [  5.0,   6.0]],
                 [[ 23.0,  24.0],
                  [ 25.0,  26.0]]]
    expectedM = [[[  7.0,   8.0],
                  [  9.0,  10.0]],
                 [[ 27.0,  28.0],
                  [ 29.0,  30.0]]]
    expectedR = [[ 11.0,  12.0],
                 [ 31.0,  32.0]]
    differenceFromEqual = np.max(np.abs(expectedP - p))
    if (differenceFromEqual > 1e-7):
        print "FAILED! Incorrect p! Difference: " + str(differenceFromEqual)
    else:
        print "OK"
    differenceFromEqual = np.max(np.abs(expectedS - s))
    if (differenceFromEqual > 1e-7):
        print "FAILED! Incorrect s! Difference: " + str(differenceFromEqual)
    else:
        print "OK"
    differenceFromEqual = np.max(np.abs(expectedM - m))
    if (differenceFromEqual > 1e-7):
        print "FAILED! Incorrect m! Difference: " + str(differenceFromEqual)
    else:
        print "OK"
    differenceFromEqual = np.max(np.abs(expectedR - r))
    if (differenceFromEqual > 1e-7):
        print "FAILED! Incorrect r! Difference: " + str(differenceFromEqual)
    else:
        print "OK"

    actualValues = np.repeat(np.expand_dims(np.asarray([0.0, 1.0]), axis=0), batch_size, axis=0)
    muValues = np.repeat(np.expand_dims(np.asarray([[0.0, 1.0], [0.0, -1.0]]), axis=0), batch_size, axis=0)
    sigmaValues = np.ones([batch_size, 2, 2])
    rhoValues = np.zeros([batch_size, 2])
    normalDistValues = sess.run(normalValues,
                                feed_dict = {
                                             actual: actualValues,
                                             mu: muValues,
                                             sigma: sigmaValues,
                                             rho: rhoValues})
    # 1 / (2 * pi) = 0.15915, the deviation of 2 gives 0.0215392793018486 in Octave.
    expectedValues = np.repeat(np.expand_dims(np.asarray([0.15915, 0.0215392793018486]), axis=0), batch_size, axis=0)
    differenceFromEqual = np.max(np.abs(expectedValues - normalDistValues))
    if (differenceFromEqual > 1e-5):
        print "FAILED! Incorrect normal distribution! Difference: " + str(differenceFromEqual)
    else:
        print "OK"
