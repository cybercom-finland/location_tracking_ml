#!/usr/bin/python -u

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
import math

parameters = params.parameters

print str(parameters)

positionTracks = load_data.load_data()

export_to_octave.export_to_octave(positionTracks)

trainData, testData, validationData = manage_data.divide(positionTracks)

export_to_octave.save('training.mat', 'training', trainData)

iter = 0
batch_size = parameters['batch_size']
network_model = model.create(parameters)
# Running 6 hours.
last_loss = train.train(parameters, network_model, trainData, testData, None, 60*6) # 'soccer-model')
