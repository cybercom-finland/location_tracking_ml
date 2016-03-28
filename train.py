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

def train(parameters, train):
    print('Launching training.')
    saver = tf.train.Saver(tf.all_variables())
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Keep training until reach max iterations for each target in the material
    
        # FIXME: This is still work in progress....
        for targetInd in range(23):
            print('Creating input data for the target: ' + str(targetInd))
            # Choosing the target to track
            trainingData = manage_data.makeInputForTargetInd(train, targetInd)
            print('Training target: ' + str(targetInd))
            step = 1
            iter = 0
            while step * batch_size < training_iters:
                learning_rate = learning_rate * decay;
                tf.assign(lr, learning_rate)
                iter += 1
                (batch_xs, batch_ys) = manage_data.getNextTrainingBatchSequences(trainingData, step - 1, batch_size)
                # Reshape data to get batch_size sequences of n_steps elements with n_input values
                batch_xs = batch_xs.reshape((batch_size, n_steps, n_input))
                batch_ys = batch_ys.reshape((batch_size, n_output))
                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, istate: stacked_lstm.zero_state(batch_size, tf.float32)})
                if step % display_step == 0:
                    saver.save(sess, 'soccer-model', global_step=iter)
                    # Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, istate: stacked_lstm.zero_state(batch_size, tf.float32)})
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, istate: stacked_lstm.zero_state(batch_size, tf.float32)})
                    prediction = sess.run(pred, feed_dict={x: batch_xs, istate: stacked_lstm.zero_state(batch_size, tf.float32)})
                    print "Prediction: " + str(prediction)
                    print "Reality: " + str(batch_ys)
                    print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                         ", Training Accuracy= " + "{:.5f}".format(acc) + ", Learning rate= " + "{:.5f}".format(learning_rate)
                step += 1
        print "Optimization Finished!"
        # Calculate accuracy for the test data
        test_len = batch_size # len(test) - 1
        
        testData = manage_data.makeInputForTargetInd(test, 0)
        test_xp, test_yp = manage_data.getNextTrainingBatchSequences(testData, 0, test_len)
        trivialCost = sess.run(cost, feed_dict={pred: test_xp[:,1,0,:], y: test_xp[:,0,0,:], istate: stacked_lstm.zero_state(batch_size, tf.float32)})
        print "Loss for just using the last known position as the prediction: " + str(trivialCost)
        # FIXME: This is still work in progress....
        testData = manage_data.makeInputForTargetInd(test, 0)
        test_xp, test_yp = manage_data.getNextTrainingBatchSequences(testData, 0, test_len)
    
        test_x = test_xp.reshape((test_len, n_steps, n_input))
        test_y = test_yp.reshape((test_len, n_output))
        print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y: test_y, istate: stacked_lstm.zero_state(batch_size, tf.float32)})
        prediction = sess.run(pred, feed_dict={x: test_x, istate: stacked_lstm.zero_state(batch_size, tf.float32)})
        print str(prediction)
        pylab.plot(test_xp[0,:,0,0], test_xp[0,:,0,1],
                 [test_xp[0,n_steps-1,0,0], prediction[0,0]],
                 [test_xp[0,n_steps-1,0,1], prediction[0,1]],
                 [test_xp[0,n_steps-1,0,0], test_yp[0,0]],
                 [test_xp[0,n_steps-1,0,1], test_yp[0,1]]);
        pylab.savefig('prediction.png')
