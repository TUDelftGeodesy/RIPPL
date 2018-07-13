#!/bin/bash
while :
do

  cd /home/everybody/radardb/radar_data/eurasia/netherlands/sentinel1
  #update_s1_benelux.sh
  S1A_download_from_SciHub_RSSv3.sh --user=fjvanleijen --password=stevin01 --area=benelux --Level=L1 --Product=SLC --ROI=roi_benelux.txt --download=false
  S1A_download_from_SciHub_RSSv3_graphical.sh --user=fjvanleijen --password=stevin01 --area=benelux --Level=L1 --Product=SLC --ROI=roi_benelux.txt --download=false

  cd /home/everybody/radardb/radar_data/america/brasil/sentinel1
  S1A_download_from_SciHub_RSSv3_2months.sh --user=fjvanleijen --password=stevin01 --area=saopaulo --Level=L1 --ROI=roi_saopaulo.txt --download=true

  #cd /home/everybody/radardb/radar_data/islands/iceland/sentinel1
  #S1A_download_from_SciHub_RSSv3_2months.sh --user=fjvanleijen --password=stevin01 --area=iceland --Level=L1 --Product=SLC --ROI=roi_reykjahlid.txt --project_functions=true

  cd /home/everybody/radardb/radar_data/eurasia/myanmar/sentinel1/yangon
  S1A_download_from_SciHub_RSSv3_2months.sh --user=fjvanleijen --password=stevin01 --area=yangon --Level=L1 --Product=SLC --ROI=roi_yangon.txt --download=true
  #S1A_download_from_SciHub_RSSv3_graphical.sh --user=fjvanleijen --password=stevin01 --area=yangon --Level=L1 --Product=SLC --ROI=roi_yangon.txt --project_functions=true
  
  echo update run finished at `date`
  sleep 1h
done

