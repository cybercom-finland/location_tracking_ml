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

import export_to_octave

# Dividing into training, test and validation set based on time

def toTensor(value):
    if (type(value) is list):
        return tf.pack(map(toTensor, value))
    return value;

def divide(positionTracks):
    
    print('Dividing into training, test and validation sets.')
    
    input = np.asarray(map(lambda l: list(l.itervalues()), positionTracks))
    third = len(input)/3
    # Leaving out the ball (player 0)
    train = input[0:third,1:23]
    test = input[third:third*2,1:23]
    validation = input[third*2:len(input),1:23]

    # Each of the sets have 22677 positions (x,y) for 22 players (plus ball).
    # We'll divide these into minibatches of size 20, getting 1133 full minibatches.
    return train, test, validation

# Returns a properly shifted input for tracking the given target.
def makeInputForTargetInd(data, targetInd):
    newData = np.copy(np.asarray(data))
    # Just moving the target player to the first position in the list.
    newData[:,[targetInd, 0], :] = newData[:,[0, targetInd], :]
    return newData;
    
# Returns one sequence of n_steps.
def getNextTrainingBatch(data, step, n_steps, n_peers):
    # Note that in data the target has already been shifted to the first place.
    # The peers are after that.
    
    # A random displacement to take the batch from.
    disp = random.randint(1, len(data[:]) - n_steps - 1)
    # Data shape: (22677, 22, 2)
    Xtrack = np.copy(np.array(data[disp:disp+n_steps,:,:]))
    # Velocity is delta to the previous position.

    Vtrack = data[disp:disp+n_steps,:,:] - data[disp-1:disp+n_steps-1,:,:]
    # We will predict the absolute position because otherwise the delta error will just accumulate
    # in the generation mode, resulting in players running out of the field.
    Ytrack = data[disp+n_steps,0,:]
    VYtrack = data[disp+n_steps,0,:] - data[disp+n_steps-1,0,:]

    targetY = Ytrack
    #targetY = np.concatenate((Ytrack, VYtrack), axis=0)
    
    #batch_input = np.concatenate((Xtrack, Vtrack), axis=2)
    batch_input = Xtrack
    
    # We will select n_peers random peers and leave out the rest.
    newPeerIndex = 1
    # All time indices for the target to track.
    final_batch = np.copy(batch_input[:,0:newPeerIndex,:])
    for peer in random.sample(range(1,22), n_peers):
        # Taking the beginning and concatenating the data of the next selected peer in the input dimension.
        selected_peer = np.copy(batch_input[:,peer:peer+1,:])
        final_batch = np.concatenate((final_batch, selected_peer), axis=1)
        newPeerIndex += 1
    
    return final_batch, targetY

def getNextTrainingBatchSequences(data, step, seqs, n_steps, n_peers):
    resultX = []
    resultY = []
    for seq in range(seqs):
        # Data is here a list of time, player, (x,y)
        sequenceX, sequenceY = getNextTrainingBatch(data, step, n_steps, n_peers)
        resultX.append(sequenceX)
        resultY.append(sequenceY)
    x = np.asarray(resultX)
    y = np.asarray(resultY)
    return x, y
