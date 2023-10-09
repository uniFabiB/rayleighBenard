#!/bin/bash

# get active directory 
currentDir=$"$PWD"
currentDir=${currentDir//" "/"\ "}

echo -n mpiexec -n 8 python3 rb.py | xclip -selection clipboard

ssh -K bxxXXXX@compu5.math.uni-hamburg.de -t "cd ${currentDir} && ./source.sh"
