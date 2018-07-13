#!/bin/sh
cd /home/everybody/orbits/radarsat2_mda &&
wget -nv -nd -r -l1 -nc ftp://r2repositoryftp:'70!7Brtp'@ftp.mda.ca/Repository/OrbitDataServer/Definitive/
# --no-verbose --no-directories --recursive --level=1 --no-clobber
# all files in one dir, only update
