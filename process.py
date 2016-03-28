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
    'learning_rate': 0.003,
    'training_iters': 1000,
    'display_step': 100,
    'decay': 0.999994,
    'input_layer': 8,
    'lstm_layers': [8, 8],
    'output_layer': 8,
    
    # Network Parameters
    # x, y for 3 targets
    # TODO: Add enabled flag
    'n_input': 3*4,
    # The minibatch is 10 sequences of 5 steps.
    'batch_size': 10,
    'n_steps': 5, # timesteps
    # x, y for 1 target. TODO: Add enabled flag.
    'n_output': 2,
    # How many peers to use in addition to the tracked target
    'n_peers': 2
}

positionTracks = load_data.load_data()

export_to_octave.export_to_octave(positionTracks)

trainData, testData, validationData = manage_data.divide(positionTracks)

network_model = model.create(parameters)

train.train(parameters, network_model, trainData, testData)
