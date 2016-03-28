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

def export_to_octave(positionTracks):

    print('Creating Octave file.')
    
    # positionTracks now has a list of all time slices, each having a dict of players containing x and y coordinates.
    octaveInput = 'pos = [\n'
    first = True;
    count = len(positionTracks)
    for state in positionTracks:
        numberOfPlayers = len(state.keys());
        for id in state.keys():
            if not first:
                octaveInput += ',\n'
            first = False
            octaveInput += ','.join(map(str, state[id]))
    octaveInput += '\n'
    octaveInput += '];\n'
    octaveInput += 'count = ' + str(count) + '\n'
    octaveInput += 'numberOfPlayers = ' + str(numberOfPlayers) + '\n'
    # The ordering of indices is funny here, because the matrix consists of position vectors.
    octaveInput += 'pos = reshape(pos, ' + str(numberOfPlayers) + ', ' + str(count) + ', 2);\n'
    with open('tracks.m', 'w') as octaveFile:
        octaveFile.write(octaveInput)
