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

import manage_data
import export_to_octave

def train(parameters, model, trainData, testingData):
    print('Launching training.')
    init = tf.initialize_all_variables()
    saver = tf.train.Saver(tf.all_variables())
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Keep training until reach max iterations for each target in the material
    
        # FIXME: This is still work in progress....
        iter = 1
        for targetInd in range(23):
            print('Creating input data for the target: ' + str(targetInd))
            # Choosing the target to track
            trainingData = manage_data.makeInputForTargetInd(trainData, targetInd)
            print('Training target: ' + str(targetInd))
            step = 1
            while step * parameters['batch_size'] < parameters['training_iters']:
                parameters['learning_rate'] = parameters['learning_rate'] * parameters['decay'];
                tf.assign(model['lr'], parameters['learning_rate'])
                (batch_xs, batch_ys) = manage_data.getNextTrainingBatchSequences(trainingData, step - 1,
                    parameters['batch_size'], parameters['n_steps'])
                
                export_to_octave.save('batch_xs_before_reshape.mat', 'batch_xs_before_reshape', batch_xs)
                # Reshape data to get batch_size sequences of n_steps elements with n_input values
                batch_xs = batch_xs.reshape((parameters['batch_size'], parameters['n_steps'], parameters['n_input']))
                batch_ys = batch_ys.reshape((parameters['batch_size'], parameters['n_output']))
                # Fit training using batch data
                sess.run(model['optimizer'], feed_dict={model['x']: batch_xs, model['y']: batch_ys,
                    model['istate']: np.asarray(model['rnn_cell'].zero_state(parameters['batch_size'],
                                                                             tf.float32).eval())})
                if step % parameters['display_step'] == 0:
                    saver.save(sess, 'soccer-model', global_step=iter)
                    
                    # For debugging, exporting a couple of arrays to Octave.
                    export_to_octave.save('batch_xs.mat', 'batch_xs', batch_xs)
                    export_to_octave.save('batch_ys.mat', 'batch_ys', batch_ys)
                    
                    # Calculate batch error as mean distance
                    error = sess.run(model['error'], feed_dict={model['x']: batch_xs, model['y']: batch_ys,
                        model['istate']: np.asarray(model['rnn_cell'].zero_state(parameters['batch_size'],
                                                                                 tf.float32).eval())})
                    # Calculate batch loss
                    loss = sess.run(model['cost'], feed_dict={model['x']: batch_xs, model['y']: batch_ys,
                        model['istate']: np.asarray(model['rnn_cell'].zero_state(parameters['batch_size'],
                                                                                 tf.float32).eval())})
                    prediction = sess.run(model['pred'], feed_dict={model['x']: batch_xs,
                        model['istate']: np.asarray(model['rnn_cell'].zero_state(parameters['batch_size'],
                                                                                 tf.float32).eval())})
                    export_to_octave.save('prediction.mat', 'prediction', prediction)
                    print "Prediction: " + str(prediction)
                    print "Reality: " + str(batch_ys)
                    print "Iter " + str(iter * parameters['batch_size']) + ", Minibatch Loss= " + \
                        "{:.6f}".format(loss) + \
                        ", Training Error= " + "{:.5f}".format(error) + ", Learning rate= " + \
                        "{:.5f}".format(parameters['learning_rate'])

                    test_len = parameters['batch_size']
                    
                    # FIXME: This is still work in progress....
                    testData = manage_data.makeInputForTargetInd(testingData, 0)
                    test_xp, test_yp = manage_data.getNextTrainingBatchSequences(testData, 0, test_len,
                                                                                 parameters['n_steps'])
                
                    test_x = test_xp.reshape((test_len, parameters['n_steps'], parameters['n_input']))
                    test_y = test_yp.reshape((test_len, parameters['n_output']))
                    export_to_octave.save('test_xp.mat', 'test_xp', test_x)
                    export_to_octave.save('test_yp.mat', 'test_yp', test_y)
                    print "Testing Error:", sess.run(model['error'], feed_dict={model['x']: test_x,
                                                                                model['y']: test_y,
                        model['istate']: np.asarray(model['rnn_cell'].zero_state(parameters['batch_size'],
                                         tf.float32).eval())})
                    prediction = sess.run(model['pred'], feed_dict={model['x']: test_x,
                        model['istate']: np.asarray(model['rnn_cell'].zero_state(parameters['batch_size'],
                                                                                 tf.float32).eval())})
                    export_to_octave.save('test_prediction.mat', 'test_prediction', prediction)
                    pylab.clf()
                    pylab.plot(test_xp[0,:,0,0], test_xp[0,:,0,1],
                             [test_xp[0,parameters['n_steps']-1,0,0], prediction[0,0]],
                             [test_xp[0,parameters['n_steps']-1,0,1], prediction[0,1]],
                             [test_xp[0,parameters['n_steps']-1,0,0], test_yp[0,0]],
                             [test_xp[0,parameters['n_steps']-1,0,1], test_yp[0,1]]);
                    pylab.savefig('prediction' + str(step) + '.png')

                iter += 1
                step += 1
        print "Optimization Finished!"
        # Calculate accuracy for the test data
