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
    
    input = map(lambda l: list(l.itervalues()), positionTracks)
    third = len(input)/3
    train = input[0:third]
    test = input[third:third*2]
    validation = input[third*2:len(input)]

    # Each of the sets have 22677 positions (x,y) for 23 players.
    # We'll divide these into minibatches of size 20, getting 1133 full minibatches.
    return train, test, validation

# Returns a properly shifted input for tracking the given target.
def makeInputForTargetInd(data, targetInd):
    newData = list(data)
    # Just moving the target player to the first position in the list.
    newData = [newData[:][targetInd]] + [x for i,x in enumerate(newData) if i != targetInd]
    return newData;
    
# Returns one sequence of n_steps.
def getNextTrainingBatch(data, step, n_steps, n_peers):
    # Note that in data the target has already been shifted to the first place.
    # The peers are after that.
    
    # A random displacement to take the batch from.
    disp = random.randint(1, len(data[:]) - n_steps - 1)
    Xtrack = np.array(data[disp:disp+n_steps])
    # Velocity is delta to the previous position.
    Vtrack = np.array(np.array(data[disp:disp+n_steps])-np.array(data[disp-1:disp+n_steps-1]))
    Ytrack = np.array(data[disp+n_steps])[0,:]
    
    batch_input = np.concatenate((Xtrack, Vtrack), axis=2)
    
    # We will select n_peers random peers and leave out the rest.
    newPeerIndex = 1
    # All time indices for the target to track.
    final_batch = batch_input[:,0:newPeerIndex,:]
    for peer in random.sample(range(1,23), n_peers):
        # Taking the beginning and concatenating the data of the next selected peer in the input dimension.
        selected_peer = batch_input[:,peer:peer+1,:]
        final_batch = np.concatenate((final_batch, selected_peer), axis=1)
        newPeerIndex += 4
    
    return final_batch, Ytrack

def getNextTrainingBatchSequences(data, step, seqs, n_steps, n_peers):
    resultX = []
    resultY = []
    for seq in range(seqs):
        # Data is here a list of time, player, (x,y)
        sequenceX, sequenceY = getNextTrainingBatch(data, step, n_steps, n_peers)
        resultX.append(sequenceX);
        resultY.append(sequenceY);
    x = np.asarray(resultX)
    y = np.asarray(resultY)
    return x, y