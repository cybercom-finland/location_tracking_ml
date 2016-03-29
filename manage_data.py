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

# Dividing into training, test and validation set based on time

def toTensor(value):
    if (type(value) is list):
        return tf.pack(map(toTensor, value))
    return value;

def divide(positionTracks):
    
    print('Dividing into training, test and validation sets.')
    
    # Taking only 100000 first ones to save memory.
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
def getNextTrainingBatch(data, step, n_steps):
    disp = random.randint(1, len(data[:]) - n_steps - 1)
    Xtrack = np.array(data[disp:disp+n_steps])
    # Velocity is delta to the previous position.
    Vtrack = np.array(np.array(data[disp:disp+n_steps])-np.array(data[disp-1:disp+n_steps-1]))
    Ytrack = np.array(data[disp+n_steps])[0,:]
    #pylab.plot(Xtrack[:,0,0], Xtrack[:,0,1], [Xtrack[n_steps-1,0,0], Ytrack[0]], [Xtrack[n_steps-1,0,1], Ytrack[1]])
    #pylab.show()
    return np.vstack((Xtrack, Vtrack)), Ytrack

def getNextTrainingBatchSequences(data, step, seqs, n_steps):
    resultX = []
    resultY = []
    for seq in range(seqs):
        sequenceX, sequenceY = getNextTrainingBatch(data, step, n_steps)
        resultX.append(sequenceX);
        resultY.append(sequenceY);
    return np.asarray(resultX), np.asarray(resultY)
