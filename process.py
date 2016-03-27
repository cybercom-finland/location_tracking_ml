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
learning_rate = 0.004
training_iters = 100000
display_step = 10
decay = 0.99995

# Network Parameters
# x, y for 23 targets
# TODO: Add velocity, enabled flag
n_input = 23*2
# The minibatch is 10 sequences of 5 steps.
batch_size = 20;
n_steps = 5 # timesteps
n_hidden = 32 # hidden layer num of features: Linear
n_hidden2 = 8 # 2. hidden layer num of features: LSTM
n_hidden3 = 4 # 3. hidden layer num of features: LSTM
# x, y for 1 target. TODO: Add enabled flag.
n_output = 2

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_output])
lstm_state_size = 2 * n_hidden2 + 2 * n_hidden3
istate = tf.placeholder("float", [None, lstm_state_size])
lr = tf.Variable(learning_rate, trainable=False)

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'hidden2': tf.Variable(tf.random_normal([n_hidden, n_hidden2])), # 2. Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden3, n_output]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'hidden2': tf.Variable(tf.random_normal([n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Returns a properly shifted input for tracking the given target.
def makeInputForTargetInd(data, targetInd):
    newData = list(data)
    newData[:][0], newData[:][targetInd] = newData[:][targetInd], newData[:][0]
    
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

def RNN(_X, _weights, _biases, states):
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # 1. hidden layer, linear activation for each batch and step.
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
    # 2. hidden layer, linear activation for each batch and step.
    _X = tf.matmul(_X, _weights['hidden2']) + _biases['hidden2']

    # Define a stacked lstm with tensorflow
    stacked_lstm = rnn_cell.MultiRNNCell([
        rnn_cell.BasicLSTMCell(n_hidden2),
        rnn_cell.BasicLSTMCell(n_hidden3, input_size=n_hidden2)])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden2)

    outputs = None
    for i in range(n_steps):
        # LSTM reuses the weights and state between steps.
        if i > 0: tf.get_variable_scope().reuse_variables()
        # The value of state is updated after processing each batch of words.
        # Get lstm output
        outputs, states = stacked_lstm(_X[i], states)

    # Get inner loop last output
    return (tf.matmul(outputs, _weights['out']) + _biases['out'], states)

pred, lastState = RNN(x, weights, biases, istate)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.l2_loss(pred-y)) # L2 loss for regression
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost) # Adam Optimizer

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
        iter = 0
        while step * batch_size < training_iters:
            learning_rate = learning_rate * decay;
            tf.assign(lr, learning_rate)
            iter += 1
            (batch_xs, batch_ys) = getNextTrainingBatchSequences(trainingData, step - 1, batch_size)
            # Reshape data to get batch_size sequences of n_steps elements with n_input values
            batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
            batch_ys = batch_ys.reshape((batch_size, n_output))
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, lstm_state_size))})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, lstm_state_size))})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, lstm_state_size))})
                prediction = sess.run(pred, feed_dict={x: batch_xs, istate: np.zeros((batch_size, lstm_state_size))})
                print "Prediction: " + str(prediction)
                print "Reality: " + str(batch_ys)
                print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                     ", Training Accuracy= " + "{:.5f}".format(acc) + ", Learning rate= " + "{:.5f}".format(learning_rate)
            step += 1
    print "Optimization Finished!"
    # Calculate accuracy for the test data
    test_len = batch_size # len(test) - 1
    
    testData = makeInputForTargetInd(test, 0)
    test_xp, test_yp = getNextTrainingBatchSequences(testData, 0, test_len)
    trivialCost = sess.run(cost, feed_dict={pred: test_xp[:,1,0,:], y: test_xp[:,0,0,:], istate: np.zeros((batch_size, lstm_state_size))})
    print "Loss for just using the last known position as the prediction: " + str(trivialCost)
    # FIXME: This is still work in progress....
    testData = makeInputForTargetInd(test, 0)
    test_xp, test_yp = getNextTrainingBatchSequences(testData, 0, test_len)

    test_x = test_xp.reshape((test_len, n_steps, n_input))
    test_y = test_yp.reshape((test_len, n_output))
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y: test_y, istate: np.zeros((batch_size, lstm_state_size))})
    prediction = sess.run(pred, feed_dict={x: test_x, istate: np.zeros((batch_size, lstm_state_size))})
    print str(prediction)
    plt.plot(test_xp[0,:,0,0], test_xp[0,:,0,1],
             [test_xp[0,n_steps-1,0,0], prediction[0,0]],
             [test_xp[0,n_steps-1,0,1], prediction[0,1]],
             [test_xp[0,n_steps-1,0,0], test_yp[0,0]],
             [test_xp[0,n_steps-1,0,1], test_yp[0,1]]);
    plt.show()
