#!/usr/bin/python

import json

# {<id>: [[x, y], [x, y], ...]}
positionTracks = dict()

# The file includes JSON lines. Don't ask me why.
with open('tr-ft.json', 'r') as inputData:
    for line in inputData:
        datum = json.loads(line)
