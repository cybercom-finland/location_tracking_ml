This is a PoC application for using Google TensorFlow and LSTM neural networks to learn
a generative neural network for multiple human location data.

This generated data can be used in simulation environments to simulate human behavior.

The goal is to take into account the spatial flow, temporal dynamics and cooperative group behavior.

Dataset
=======

The dataset used is player location data from a soccer game.
The source for the dataset is https://datahub.io/dataset/magglingen2013, the game TR vs. FT.

Run `./getData.sh` to get the dataset.

Run `./process.py` to parse the dataset and convert it into Octave format for inspection.

The coordinates x and y are limited by the soccer field size (assumption here): x = [-53, 53], y = [-35, 35].

You can plot the data in Octave using:

`tracks`

`p = 5;`

`indices = (abs(pos(p,1,:)) <= 53 & abs(pos(p,2,:)) <= 35); scatter(pos(p,1,indices), pos(p,2,indices), 2, p);`

Here 5 is the index of the player, and the first command filters out the empty and otherwise invalid values.

Screenshots of sample data
=========================

![example_track.png](example_track.png)

![first_5_tracks.png](first_5_tracks.png)

Neural network structure and intuition
======================================

The neural network should comprise of modules, each estimating the next position of a particular
target. Exploiting the symmetries in the domain, all modules have identical weights.
The inputs (and internal states for RNNs) differ per module respecting the special position for the self-input
corresponding to the input for the target for which this module is predicting the next position.

These modules should be recurrent to have time dynamics in addition to trivial average flow in field space.

The weights should be forced equal so that each module is interchangeable so that:
 * The associated weights towards self-input are the same for all modules.
 * The associated weights towards inputs other than self are identical.

Therefore, the model of one target is used for all targets.
The model can learn coordination between targets, that is, formations and complex team dynamics,
because the modules get information of the positions of other targets as well.

LSTM modules will be used for this exercise.

The intuition behind layers is that the first LSTM layer models the trivial space-time domain kinematics
of the target, and relations to peers, and trivial flocking behavior, see Boids.
The second LSTM layer models the higher level coordination dynamics.

Trying simple things first, going towards more complex ideas to evaluate improvement. The number and type of layers will
change as a result of experiments later.

Coding
======

The locations are coded as simple x and y floating point values, as in the original data.
Each player has a pair of input neurons corresponding to the current position, and the last position before that
for both coordinates x and y. For one module and 23 players, this makes 23x2x2 = 92 continuous valued input neurons
for each module. The input data is the same for each module, but the target to predict is shifted to the first neurons.

The value signifying missing values is replaced by a special on-off neuron. This neuron is off when either of the input
values is the placeholder value signifying missing data. When the neuron is off, both x and y neurons respectively are
zeroed.

The output neuron coding is identical to the input neuron coding but only has x and y positions per module.
In generation mode the output can be fed back to the inputs by calculating the deltas.

Ideas
=====

 * It might make sense to encode the locations of other targets in the order of distance, and using delta coding instead
of absolute x and y. Then the non-self targets do not need equal weights.
