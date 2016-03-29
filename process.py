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

import load_data
import export_to_octave
import manage_data
import model
import train

# TODO: Make a run with different parameters and plot results
parameters = {
    'learning_rate': 0.005,
    'training_iters': 10000,
    'display_step': 100,
    'decay': 0.9999,
    'input_layer': 12,
    'lstm_layers': [8],
    'output_layer': 8,
    # How many targets are there
    'n_targets': 23,
    # x, y for 3 targets
    # TODO: Add enabled flag
    'n_input': 23*4,
    # The minibatch is 10 sequences of 5 steps.
    'batch_size': 10,
    'n_steps': 5, # timesteps
    # x, y for 1 target. TODO: Add enabled flag.
    'n_output': 2
}

print str(parameters)

positionTracks = load_data.load_data()

export_to_octave.export_to_octave(positionTracks)

trainData, testData, validationData = manage_data.divide(positionTracks)

network_model = model.create(parameters)

train.train(parameters, network_model, trainData, testData)
