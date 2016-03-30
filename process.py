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
    'learning_rate': 0.01,
    'training_iters': 10000,
    'display_step': 10,
    'decay': 0.99995,
    'input_layer': None,
    'lstm_layers': [16],
    # How many targets are there
    'n_targets': 23,
    'n_peers': 2,
    # x, y for 3 targets
    # TODO: Add enabled flag
    'n_input': 3*4,
    # The minibatch is 16 sequences of 5 steps.
    'batch_size': 16,
    'n_steps': 5, # timesteps
    # x, y for 1 target. TODO: Add enabled flag.
    'n_output': 2,
    'lstm_clip': 10.0
}

print str(parameters)

positionTracks = load_data.load_data()

export_to_octave.export_to_octave(positionTracks)

trainData, testData, validationData = manage_data.divide(positionTracks)

export_to_octave.save('training.mat', 'training', trainData)

network_model = model.create(parameters)

train.train(parameters, network_model, trainData, testData)
