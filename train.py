#!/usr/bin/python

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
import math

import manage_data
import export_to_octave

import time

def train(parameters, model, trainData, testingData, start, minutes):
    print('Launching training.')
#    accuracy_summary = tf.scalar_summary("cost", model["cost"])
#    merged = tf.merge_all_summaries()
    init = tf.initialize_all_variables()
    saver = tf.train.Saver(tf.all_variables())
    # Launch the graph
    # config=tf.ConfigProto(log_device_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    start_time = time.time()
    with tf.Session() as sess:
        if start:
            saver.restore(sess, start)
        else:
            sess.run(init)
    #    pred = (tf.Print(pred[0], [tf.reduce_min(tf.abs(variable))], "Var " + variable.name + " zero dist: "), pred[1])
    #    pred = (tf.Print(pred[0], [tf.reduce_max(variable)], "Var " + variable.name + " max: "), pred[1])
    #    pred = (tf.Print(pred[0], [tf.reduce_min(variable)], "Var " + variable.name + " min: "), pred[1])
        
#        writer = tf.train.SummaryWriter("logs", sess.graph)
        # Keep training until reach max iterations for each target in the material
    
        iter = 1
        
        step = 1
        trainErrorTrend = []
        testErrorTrend = []
        now = time.time()
        # Training for a specific number of minutes
        last_losses = []
        last_loss = None
        while now - start_time < 60 * minutes:
            targetInd = random.randint(0, 21)
            print('Creating input data for the target: ' + str(targetInd))
            if last_loss:
                print "Time elapsed: ", now - start_time, ", last_loss: ", last_loss / parameters['batch_size']
            # Choosing the target to track
            trainingData = manage_data.makeInputForTargetInd(trainData, targetInd)
            #export_to_octave.save('training_data_d.mat', 'trainingData', trainingData)
        
            # parameters['learning_rate'] = parameters['learning_rate'] * parameters['decay']
            (batch_xsp, batch_ysp) = manage_data.getNextTrainingBatchSequences(trainingData, step - 1,
                parameters['batch_size'], parameters['n_steps'], parameters['n_peers'])
            
            #export_to_octave.save('batch_xs_before_reshape.mat', 'batch_xs_before_reshape', batch_xsp)
            # Reshape data to get batch_size sequences of n_steps elements with n_input values
            batch_xs = batch_xsp.reshape((parameters['batch_size'], parameters['n_steps'], parameters['n_input']))
            batch_ys = batch_ysp.reshape((parameters['batch_size'], parameters['n_output']))

            # Saving the previous good model.
            if step % parameters['display_step'] == 0:
                saver.save(sess, 'soccer-model-prev')
            
            # Fit training using batch data
            ## 'optimizer'
            sess.run([model['optimizer']], feed_dict = {
                model['x']: batch_xs,
                model['y']: batch_ys,
                model['istate']: np.asarray(model['rnn_cell'].zero_state(parameters['batch_size'],
                                                                         tf.float32).eval()),
                model['lr']: parameters['learning_rate'],
                model['keep_prob']: parameters['keep_prob']
            })
            if step % parameters['display_step'] == 0:
                saver.save(sess, 'soccer-model')
                testData = manage_data.makeInputForTargetInd(testingData, np.random.randint(0, 22))
                test_len = parameters['batch_size']
                
                testTarget = random.randint(0, 21)
                predictedBatch = random.randint(0, test_len-1)
                
                
                # For debugging, exporting a couple of arrays to Octave.
                #export_to_octave.save('batch_xs.mat', 'batch_xs', batch_xs)
                #export_to_octave.save('batch_ys.mat', 'batch_ys', batch_ys)
                
                # Calculate batch error as mean distance
                [error, prediction] = sess.run([tf.stop_gradient(model['cost']), tf.stop_gradient(model['pred'])], feed_dict = {
                    model['x']: batch_xs,
                    model['y']: batch_ys,
                    model['istate']: np.asarray(model['rnn_cell'].zero_state(parameters['batch_size'],
                                                                             tf.float32).eval()),
                    model['keep_prob']: parameters['keep_prob']
                })
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
                    #"{:.6f}".format(error) + \
                print "Iter " + str(iter * parameters['batch_size']) + ", Minibatch Loss= " + \
                    "{:.6f}".format(error) + \
                    ", Learning rate= " + \
                    "{:.5f}".format(parameters['learning_rate'])

                test_xp, test_yp = manage_data.getNextTrainingBatchSequences(testData, testTarget, test_len,
                                                                             parameters['n_steps'],
                                                                             parameters['n_peers'])

                test_x = test_xp.reshape((test_len, parameters['n_steps'], parameters['n_input']))
                test_y = test_yp.reshape((test_len, parameters['n_output']))
                #export_to_octave.save('test_xp.mat', 'test_xp', test_x)
                #export_to_octave.save('test_yp.mat', 'test_yp', test_y)
                [testError, prediction] = sess.run([tf.stop_gradient(model['cost']), tf.stop_gradient(model['pred'])],
                    feed_dict={model['x']: test_x,
                    model['y']: test_y,
                    model['istate']: np.asarray(model['rnn_cell'].zero_state(parameters['batch_size'],
                                     tf.float32).eval()),
                                                    model['keep_prob']: parameters['keep_prob']})
                ##writer.add_summary(summary_str, iter)
                testErrorTrend.append(testError)
                last_losses.append(testError)
                # Averaging the 3 last testing losses.
                if (len(last_losses) > 3):
                    last_losses.pop(0)
                last_loss = math.fsum(last_losses) / len(last_losses)
                print "Testing Error:", testError
                print "Testing Error Normalized:", testError / parameters['batch_size']
                print "Last loss:", last_loss / parameters['batch_size']
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
            now = time.time()
        saver.save(sess, 'soccer-model', global_step=iter)
        print "Optimization Finished!"
        
        # Returning the last loss value for hyper parameter search
        return last_loss
    