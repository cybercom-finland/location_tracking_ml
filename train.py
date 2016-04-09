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
    accuracy_summary = tf.scalar_summary("cost", model["cost"])
    merged = tf.merge_all_summaries()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver(tf.all_variables())
    # Launch the graph
    with tf.Session() as sess:
        writer = tf.train.SummaryWriter("logs", sess.graph_def)
        sess.run(init)
        # Keep training until reach max iterations for each target in the material
    
        # FIXME: This is still work in progress....
        iter = 1
        trainErrorTrend = []
        testErrorTrend = []
        
        for targetInd in range(0, 22):
            print('Creating input data for the target: ' + str(targetInd))
            # Choosing the target to track
            trainingData = manage_data.makeInputForTargetInd(trainData, targetInd)
            #export_to_octave.save('training_data_d.mat', 'trainingData', trainingData)
            
            step = 1
            while step * parameters['batch_size'] < parameters['training_iters']:
                parameters['learning_rate'] = parameters['learning_rate'] * parameters['decay']
                tf.assign(model['lr'], parameters['learning_rate'])
                (batch_xsp, batch_ysp) = manage_data.getNextTrainingBatchSequences(trainingData, step - 1,
                    parameters['batch_size'], parameters['n_steps'], parameters['n_peers'])
                
                #export_to_octave.save('batch_xs_before_reshape.mat', 'batch_xs_before_reshape', batch_xsp)
                # Reshape data to get batch_size sequences of n_steps elements with n_input values
                batch_xs = batch_xsp.reshape((parameters['batch_size'], parameters['n_steps'], parameters['n_input']))
                batch_ys = batch_ysp.reshape((parameters['batch_size'], parameters['n_output']))
                # Fit training using batch data
                sess.run(model['optimizer'], feed_dict={model['x']: batch_xs, model['y']: batch_ys,
                    model['istate']: np.asarray(model['rnn_cell'].zero_state(parameters['batch_size'],
                                                                             tf.float32).eval())})
                if step % parameters['display_step'] == 0:
                    
                    testData = manage_data.makeInputForTargetInd(testingData, np.random.randint(0, 22))
                    test_len = parameters['batch_size']
                    
                    testTarget = random.randint(0, 22)
                    predictedBatch = random.randint(0, test_len-1)
                    
                    saver.save(sess, 'soccer-model', global_step=iter)
                    
                    # For debugging, exporting a couple of arrays to Octave.
                    #export_to_octave.save('batch_xs.mat', 'batch_xs', batch_xs)
                    #export_to_octave.save('batch_ys.mat', 'batch_ys', batch_ys)
                    
                    # Calculate batch error as mean distance
                    [error, prediction, loss] = sess.run([model['cost'], model['pred'], model['cost']],
                        feed_dict={model['x']: batch_xs, model['y']: batch_ys,
                        model['istate']: np.asarray(model['rnn_cell'].zero_state(parameters['batch_size'],
                                                                                 tf.float32).eval())})
                    trainErrorTrend.append(error)

                    print "Test target: " + str(testTarget)
                    print "Batch: " + str(predictedBatch)
                    if (False):
                        pylab.clf()
                        pylab.plot(batch_xsp[predictedBatch,:,0,0], batch_xsp[predictedBatch,:,0,1],
                             [batch_xsp[predictedBatch,parameters['n_steps']-1,0,0],
                              batch_xsp[predictedBatch,parameters['n_steps']-1,0,0] +
                                  prediction[predictedBatch,0]],
                             [batch_xsp[predictedBatch,parameters['n_steps']-1,0,1],
                              batch_xsp[predictedBatch,parameters['n_steps']-1,0,1] +
                                  prediction[predictedBatch,1]],
                             [batch_xsp[predictedBatch,parameters['n_steps']-1,0,0],
                              batch_xsp[predictedBatch,parameters['n_steps']-1,0,0] +
                                  batch_ysp[predictedBatch,0]],
                             [batch_xsp[predictedBatch,parameters['n_steps']-1,0,1],
                              batch_xsp[predictedBatch,parameters['n_steps']-1,0,1] +
                                  batch_ysp[predictedBatch,1]]);
                        pylab.savefig('prediction_train_' + str(iter) + '.png')
                    
                    # Calculate batch loss
                    n_mixtures = parameters['n_mixtures']
                    #export_to_octave.save('prediction.mat', 'prediction', prediction)
                    # Printing out mus.
                    print "Weights: " + str(prediction[:, 0:n_mixtures])
                    print "Prediction sigmas: " + str(prediction[:, n_mixtures : n_mixtures * 3])
                    print "Prediction mus: " + str(prediction[:, n_mixtures * 3 : n_mixtures * 5])
                    print "Prediction rhos: " + str(prediction[:, n_mixtures * 5 : n_mixtures * 6])
                    print "Reality: " + str(batch_ys)
                    print "Iter " + str(iter * parameters['batch_size']) + ", Minibatch Loss= " + \
                        "{:.6f}".format(loss) + \
                        ", Training Error= " + "{:.5f}".format(error) + ", Learning rate= " + \
                        "{:.5f}".format(parameters['learning_rate'])

                    test_xp, test_yp = manage_data.getNextTrainingBatchSequences(testData, testTarget, test_len,
                                                                                 parameters['n_steps'],
                                                                                 parameters['n_peers'])

                    test_x = test_xp.reshape((test_len, parameters['n_steps'], parameters['n_input']))
                    test_y = test_yp.reshape((test_len, parameters['n_output']))
                    #export_to_octave.save('test_xp.mat', 'test_xp', test_x)
                    #export_to_octave.save('test_yp.mat', 'test_yp', test_y)
                    [summary_str, testError, prediction] = sess.run([merged, model['cost'], model['pred']],
                        feed_dict={model['x']: test_x,
                        model['y']: test_y,
                        model['istate']: np.asarray(model['rnn_cell'].zero_state(parameters['batch_size'],
                                         tf.float32).eval())})
                    writer.add_summary(summary_str, iter)
                    testErrorTrend.append(testError)
                    print "Testing Error:", testError
                    export_to_octave.save('train_error.mat', 'train_error', trainErrorTrend)
                    export_to_octave.save('test_error.mat', 'test_error', testErrorTrend)
                    #export_to_octave.save('test_prediction.mat', 'test_prediction', prediction)
                    if (False):
                        pylab.clf()
                        pylab.plot(test_xp[predictedBatch,:,0,0], test_xp[predictedBatch,:,0,1],
                             [test_xp[predictedBatch,parameters['n_steps']-1,0,0],
                              prediction[predictedBatch,0]],
                             [test_xp[predictedBatch,parameters['n_steps']-1,0,1],
                              prediction[predictedBatch,1]],
                             [test_xp[predictedBatch,parameters['n_steps']-1,0,0],
                              test_yp[predictedBatch,0]],
                             [test_xp[predictedBatch,parameters['n_steps']-1,0,1],
                              test_yp[predictedBatch,1]]);
                        pylab.savefig('prediction' + str(iter) + '.png')

                iter += 1
                step += 1
        saver.save(sess, 'soccer-model', global_step=iter)
        print "Optimization Finished!"
        # Calculate accuracy for the test data
