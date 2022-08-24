#!/bin/bash
avgCellSize=0.01
#
#
#
minFactor=0.9
maxFactor=1.1
minCellSize=`echo "scale=scale($avgCellSize)+2;var=$avgCellSize;var*=$minFactor;var" | bc`
maxCellSize=`echo "scale=scale($avgCellSize)+2;var=$avgCellSize;var*=$maxFactor;var" | bc`
gmsh mygeo.geo -2 -format msh2 -clmin $minCellSize -clmax $maxCellSize
read -n 1 -s -r -p "Press any key to continue"
