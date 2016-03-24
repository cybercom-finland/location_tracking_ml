#!/usr/bin/python

import tensorflow as tf
import json

# {<id>: [[x, y], [x, y], ...]}
positionTracks = dict()
# The file includes JSON lines. Don't ask me why.
# The rows look like this:
# {"game":"NN7SxDe8uLmvKLiQd","ts":3232387,"data":[{"id":5,"x":-1.108,"y":0.155,"distToBall":0,"poss":false},{"id":11,"x":-0.27,"y":-0.63,"distToBall":1.1482460537707064,"poss":false},{"id":12,"x":-0.408,"y":-29.365,"distToBall":29.52829829163882,"poss":false},{"id":15,"x":-45.583,"y":-0.254,"distToBall":44.47688057856576,"poss":false},{"id":16,"x":-15.173,"y":7.564,"distToBall":15.89709111755984,"poss":false},{"id":19,"x":-15.845,"y":-4.697,"distToBall":15.515188461633329,"poss":false},{"id":20,"x":-1.026,"y":29.019,"distToBall":28.86411647703771,"poss":false},{"id":22,"x":-0.014,"y":27.519,"distToBall":27.385860074133145,"poss":false},{"id":23,"x":-0.053,"y":0.547,"distToBall":1.125472789542244,"poss":false},{"id":25,"x":-9.062,"y":-16.995,"distToBall":18.904724700455176,"poss":false},{"id":29,"x":-6.073,"y":0.108,"distToBall":4.96522245221702,"poss":false},{"id":30,"x":-11.883,"y":1.116,"distToBall":10.817769918056122,"poss":false},{"id":41,"x":20.325,"y":21.94,"distToBall":30.56075447367097,"poss":false},{"id":43,"x":19.473,"y":-15.108,"distToBall":25.622972700293776,"poss":false},{"id":45,"x":21.959,"y":7.454,"distToBall":24.194253243280727,"poss":false},{"id":46,"x":14.353,"y":-2.178,"distToBall":15.636029227396579,"poss":false},{"id":47,"x":14.344,"y":6.372,"distToBall":16.65579157530497,"poss":false},{"id":60,"x":9.481,"y":0.167,"distToBall":10.589006799506741,"poss":false},{"id":61,"x":8.129,"y":24.685,"distToBall":26.211506423706364,"poss":false},{"id":64,"x":3.442,"y":-13.896,"distToBall":14.769329741054602,"poss":false},{"id":65,"x":20.937,"y":-3.572,"distToBall":22.357829814183667,"poss":false},{"id":72,"x":45.615,"y":0.148,"distToBall":46.72300052436701,"poss":false},{"id":73,"x":-0.249,"y":8.613,"distToBall":8.501508395573106,"poss":false}],"calc":{"pressingIndex":0,"curPass":0,"curBPTimeTeam":{"0":{"a":0,"s":{"1":0,"2":0,"3":0}},"1":{"a":0,"s":{"1":0,"2":0,"3":0}},"2":{"a":0,"s":{"1":0,"2":0,"3":0}}}},"_id":"WTE3PjZ4eEC3cFgo7"}

# We want to change it into a list of positions, posx vs. posy
count = 0
with open('tr-ft.json', 'r') as inputData:
    # Just counting the rows first.
    count = sum(1 for line in inputData)
with open('tr-ft.json', 'r') as inputData:
    # Just collecting all the player ids first.
    for line in inputData:
        datum = json.loads(line)
        for pos in datum['data']:
            if (not positionTracks.has_key(pos['id'])):
                positionTracks[pos['id']] = [None] * count;
index = 0
with open('tr-ft.json', 'r') as inputData:
    for line in inputData:
        datum = json.loads(line)
        for pos in datum['data']:
            # Note: The data has 99999.999 for x and y denoting missing position.
            # Leaving them here, because we can't have null values in any case.
            positionTracks[pos['id']][index] = [pos['x'], pos['y']];
        index += 1

octaveInput = 'pos = [\n'
first = True;
for id in positionTracks.keys():
    if not first:
        octaveInput += ',\n'
    first = False
    octaveInput += ','.join(map(str, positionTracks[id]))
octaveInput += '\n'
octaveInput += '];\n'
numberOfPlayers = len(positionTracks.keys());
octaveInput += 'pos = reshape(pos, ' + str(numberOfPlayers) + ', ' + str(index) + ', 2);\n'
with open('tracks.m', 'w') as octaveFile:
    octaveFile.write(octaveInput)
# You can draw the track in octave with:
#  tracks
#  p = 5;
#  indices = (max(pos(p,:,1),pos(p,:,2)) < 999); scatter(pos(p,indices,1), pos(p,indices,2));
# Here 5 is the index of the player, and the first lines filter out the bad values.

# Neural network structure and intuition:
# The neural network should comprise of modules, each estimating the next position of a particular
# target.
# These modules should be recurrent to have time dynamics in addition to trivial average flow in field space.
# The weights should be forced equal so that each module is interchangeable so that:
# - The weights towards self-input are the same for all modules.
# - The weights towards inputs other than self are identical.
# Therefore, all modules learn what one module sees, so the model of one target is used for all targets.
# The model can learn coordination between targets, that is, formations and complex team dynamics,
# because the modules get information of the positions of other targets as well.

# LSTM modules with specially shared weights will be used for this exercise.
