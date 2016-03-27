#!/usr/bin/python

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import matplotlib.pyplot as plt

import random
import json
import itertools

# {<id>: [[x, y], [x, y], ...]}
positionTracks = dict()
# The file includes JSON lines. Don't ask me why.
# The rows look like this:
# {"game":"NN7SxDe8uLmvKLiQd","ts":3232387,"data":[{"id":5,"x":-1.108,"y":0.155,"distToBall":0,"poss":false},{"id":11,"x":-0.27,"y":-0.63,"distToBall":1.1482460537707064,"poss":false},{"id":12,"x":-0.408,"y":-29.365,"distToBall":29.52829829163882,"poss":false},{"id":15,"x":-45.583,"y":-0.254,"distToBall":44.47688057856576,"poss":false},{"id":16,"x":-15.173,"y":7.564,"distToBall":15.89709111755984,"poss":false},{"id":19,"x":-15.845,"y":-4.697,"distToBall":15.515188461633329,"poss":false},{"id":20,"x":-1.026,"y":29.019,"distToBall":28.86411647703771,"poss":false},{"id":22,"x":-0.014,"y":27.519,"distToBall":27.385860074133145,"poss":false},{"id":23,"x":-0.053,"y":0.547,"distToBall":1.125472789542244,"poss":false},{"id":25,"x":-9.062,"y":-16.995,"distToBall":18.904724700455176,"poss":false},{"id":29,"x":-6.073,"y":0.108,"distToBall":4.96522245221702,"poss":false},{"id":30,"x":-11.883,"y":1.116,"distToBall":10.817769918056122,"poss":false},{"id":41,"x":20.325,"y":21.94,"distToBall":30.56075447367097,"poss":false},{"id":43,"x":19.473,"y":-15.108,"distToBall":25.622972700293776,"poss":false},{"id":45,"x":21.959,"y":7.454,"distToBall":24.194253243280727,"poss":false},{"id":46,"x":14.353,"y":-2.178,"distToBall":15.636029227396579,"poss":false},{"id":47,"x":14.344,"y":6.372,"distToBall":16.65579157530497,"poss":false},{"id":60,"x":9.481,"y":0.167,"distToBall":10.589006799506741,"poss":false},{"id":61,"x":8.129,"y":24.685,"distToBall":26.211506423706364,"poss":false},{"id":64,"x":3.442,"y":-13.896,"distToBall":14.769329741054602,"poss":false},{"id":65,"x":20.937,"y":-3.572,"distToBall":22.357829814183667,"poss":false},{"id":72,"x":45.615,"y":0.148,"distToBall":46.72300052436701,"poss":false},{"id":73,"x":-0.249,"y":8.613,"distToBall":8.501508395573106,"poss":false}],"calc":{"pressingIndex":0,"curPass":0,"curBPTimeTeam":{"0":{"a":0,"s":{"1":0,"2":0,"3":0}},"1":{"a":0,"s":{"1":0,"2":0,"3":0}},"2":{"a":0,"s":{"1":0,"2":0,"3":0}}}},"_id":"WTE3PjZ4eEC3cFgo7"}

print('Reading input data file.')
# We want to change it into a list of positions for each player, posx vs. posy
count = 0
with open('tr-ft.json', 'r') as inputData:
    # Just counting the rows first.
    count = sum(1 for line in inputData)
index = 0
with open('tr-ft.json', 'r') as inputData:
    positionTracks = [None] * count;
    for line in inputData:
        datum = json.loads(line)
        positionTracks[index] = dict()
        for pos in datum['data']:
            # Note: The data has 99999.999 for x and y denoting missing position.
            # Cleaning them up by filling them with the last known good location.
            if float(pos['x']) > 1000:
                positionTracks[index][pos['id']] = positionTracks[index-1][pos['id']];
            else:
                positionTracks[index][pos['id']] = [float(pos['x']), float(pos['y'])];
        index += 1

print('Creating Octave file.')

# positionTracks now has a list of all time slices, each having a dict of players containing x and y coordinates.
octaveInput = 'pos = [\n'
first = True;
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

# Dividing into training, test and validation set based on time

def toTensor(value):
    if (type(value) is list):
        return tf.pack(map(toTensor, value))
    return value;

print('Dividing into training, test and validation sets.')

# Taking only 100000 first ones to save memory.
input = map(lambda l: list(l.itervalues()), positionTracks)
third = len(input)/3
train = input[0:third]
test = input[third:third*2]
validation = input[third*2:len(input)]

# Each of the sets have 22677 positions (x,y) for 23 players.
# We'll divide these into minibatches of size 20, getting 1133 full minibatches.

# Let's try this first with fully connected LSTM layers. Note that the implementation is not yet
# as specified in the README. This is work in progress.


# The input is position and velocity for each player. Velocity is 0 if not otherwise possible to calculate.
# TODO: Flag input for valid or invalid position.

print('Creating the neural network model.')
# Parameters
learning_rate = 0.0001
training_iters = 100000
display_step = 10

# Network Parameters
# x, y for 23 targets
# TODO: Add velocity, enabled flag
n_input = 23*2
# The minibatch is 20 sequences of 20 steps.
batch_size = 20;
n_steps = 5 # timesteps
n_hidden = 20 # hidden layer num of features
# x, y for 1 target. TODO: Add enabled flag.
n_output = 2

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
istate = tf.placeholder("float", [None, 2*n_hidden])
y = tf.placeholder("float", [None, n_output])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Returns a properly shifted input for tracking the given target.
def makeInputForTargetInd(data, targetInd):
    newData = list(data)
    newData[:][0], newData[:][targetInd] = newData[:][targetInd], newData[:][0]
    #plt.plot(data[:][0][0], data[:][0][1])
    #plt.plot(data[:][1][0], data[:][1][1])
    #plt.show()
    
    return newData;
    
# Returns one sequence of n_steps.
def getNextTrainingBatch(data, step):
    disp = random.randint(0, len(data) - n_steps - 1)
    Xtrack = np.array(data[disp:disp+n_steps])
    Ytrack = np.array(data[disp+n_steps])[0,:]
    #plt.plot(Xtrack[:,0,0], Xtrack[:,0,1], [Xtrack[n_steps-1,0,0], Ytrack[0]], [Xtrack[n_steps-1,0,1], Ytrack[1]])
    #plt.show()
    return Xtrack, Ytrack

def getNextTrainingBatchSequences(data, step, seqs):
    resultX = []
    resultY = []
    for seq in range(seqs):
        sequenceX, sequenceY = getNextTrainingBatch(data, step)
        resultX.append(sequenceX);
        resultY.append(sequenceY);
    return np.asarray(resultX), np.asarray(resultY)

def RNN(_X, _istate, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = RNN(x, istate, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.l2_loss(pred-y)) # L2 loss for regression
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Evaluate model, 1 m accuracy is ~1.0. Higher accuracy is better.
# We will take 1 m as the arbitrary goal post to be happy with the accuracy.
accuracy = 1.0 / (tf.sqrt(tf.nn.l2_loss(pred-y) * 2) + 0.001)

# Initializing the variables
init = tf.initialize_all_variables()

print('Launching training.')
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations for each target in the material
    
    # FIXME: This is still work in progress....
    for targetInd in range(23):
        print('Creating input data for the target: ' + str(targetInd))
        # Choosing the target to track
        trainingData = makeInputForTargetInd(train, targetInd)
        print('Training target: ' + str(targetInd))
        step = 1
        while step * batch_size < training_iters:
            (batch_xs, batch_ys) = getNextTrainingBatchSequences(trainingData, step - 1, batch_size)
            # Reshape data to get batch_size sequences of n_steps elements with n_input values
            batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
            batch_ys = batch_ys.reshape((batch_size, n_output))
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                           istate: np.zeros((batch_size, 2*n_hidden))})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                    istate: np.zeros((batch_size, 2*n_hidden))})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                                 istate: np.zeros((batch_size, 2*n_hidden))})
                print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                     ", Training Accuracy= " + "{:.5f}".format(acc)
            step += 1
    print "Optimization Finished!"
    # Calculate accuracy for the test data
    test_len = len(test) - 1
    
    # FIXME: This is still work in progress....
    testData = makeInputForTargetInd(test, 0)
    test_x, test_y = getNextTrainingBatchSequences(testData, 0, test_len)
    test_x = test_x.reshape((test_len, n_steps, n_input))
    test_y = test_y.reshape((test_len, n_output))
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y: test_y,
                                                             istate: np.zeros((test_len, 2*n_hidden))})
