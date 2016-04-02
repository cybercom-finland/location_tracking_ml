#!/bin/bash
ffmpeg -pix_fmt yuv420p -framerate 10 -i "output/%5d.png" -vf fps=10 -vcodec mpeg4 -strict -2 -b:v 800k -shortest track.mp4

