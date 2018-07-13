#!/bin/bash
while :
do
  ./update_orbits_radarsat2.sh
  ./download_radarsat2_nso.sh
  python ~/bin/rsat2/update_radarsat.py
#  ./download_radarsat2_greenland.sh
#  python ~/bin/rsat2/update_radarsat_greenland.py
#  ./download_radarsat2_saopaulo.sh
#  python ~/bin/rsat2/update_radarsat_saopaulo.py
#  ./download_radarsat2_saopaulo2.sh
#  python ~/bin/rsat2/update_radarsat_saopaulo2.py
  echo update run finished at `date`
  sleep 1h
done

