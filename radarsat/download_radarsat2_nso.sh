#!/bin/sh
cd /home/fjvanleijen/new_data/nl_radarsat2_nso &&
wget -nv -nH -r -l1 -nc ftp://nsoportal:N\$oport4L@fd1.rsi.ca  2>&1| grep -v utime
# --no-verbose --no-host-directories --recursive --level=1 --no-clobber
# only update

# note: && causes next command only to be executed when previous command succeeds
