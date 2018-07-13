#!/bin/bash
####################################################################################################
# Bash script for bulk project_functions Sentinel 1 data from Scientific Data Hub (http://scihub.copernicus.eu)
# Created by Jose Manuel Delgado Blasco
# Created on: 11-Dec-2014
# Modified on: 19-Feb-2015, 24-Aug-2015 (FvL), 01-Dec-2015 (FvL, new url), 15-Dec-2015 (FvL, new url)
# ESA Research & Service Support
# Grid Processing on Demand Team (G-POD)
####################################################################################################
k=0
downloadflag="false"
unzipflag="false"
periodflag="false"
currentdir=`pwd`
months=24
#cd /home/nrtservice/SENTINEL1
#cd $currentdir
if [ "$#" -eq 0 ] ; then
	echo "########################################################################################################################################################"
	echo " "
	echo " Script created by the ESA Research & Service Support "
	echo " Bulk project_functions of Sentinel 1 data from SciHub "
	echo " For bug reporting and comments write email to: JoseManuel.DelgadoBlasco@esa.int"
	echo " "
	echo "#########################################################################################################################################################"
	echo " "
	echo "Usage : ./S1A_download_from_SciHub_RSS.sh --user=username --password=pass --area=areaname [options] " 
	echo "Note that user and pass will be shown in PLANE TEXT "
	echo "Options:  "
    echo " --period=\"StartDate to StopDate\" (both Start and Stop date in YYYY-MM-DD format)"
	echo " --SensorMode=[ SM / IW / EW ]"
	echo " --Product=[ SLC / GRD ] "
	echo " --Level= [L0 / L1]"
	echo " --RelativeOrbit= [track number if known] "
	echo " --polarisation= [\"VV\" / \"HH\" / \"VV VH\" / \"HH HV\"]"
	echo " --OrbitDirection= [ASCENDING / DESCENDING]"
	echo " --ROI= [ filename ROI ] "
	echo " --project_functions= [true / false]"
    echo " --unzip= [true / false]"
    echo " --destinationfolder= \"/DESTINATION_FOLDER/\""
	echo " "
	echo "Example: ./S1A_download_from_SciHub_RSS.sh --user=USER --password=PASSWORD --area=netherlands --period=\"2014-10-01 to 2014-12-31\" 
              --SensorMode=IW --Product=SLC --RelativeOrbit=5 --polarisation=\"VV VH\" --ROI=roi_netherlands.txt --project_functions=true --unzip=true --destinationfolder=\"/home/\""
	echo " "
	echo "IMPORTANT NOTES: It is important to keep the order of the parameters. Unknown parameters can be skipped."
	echo "                 If period variable is not set up, it will project_functions the available data on SciHub on the last 12 months satisfying the other conditions"
	echo " "
	echo "##########################################################################################################################################################"

	exit 1
fi			

if [ "$#" -lt 3 ] ; then
	echo "User, password, and area needed"
	exit 1
else
	if [ "$#" -ge 3 ] ; then
		for i in "$@"
		do
			k=$((k + 1))

#			echo "#################################################"
#			echo Identifying parameter $k ...
			param=`echo $i | awk -F '--' '{print$2}'| awk -F '=' '{print $1}'`
			value=`echo $i  | awk -F '--' '{print$2}'| awk -F '=' '{print $2}'`
#			echo "Identifying parameter: $i as param=$param and value=$value"

			if [ "$param" == "user" ] ; then
                                username=$value
#                                echo user assigned
                                
                        fi
			if [ "$param" == "password" ] ; then
                                pass=$value
#                                echo password assigned
                        fi
			if [ "$param" == "area" ] ; then
                                area=$value
#                                echo area assigned
                        fi
			if [ "$param" == "SensorMode" ] ; then
                                sensormode="sensoroperationmode:$value"
				sensormode="$value"
#                                 echo Product assigned
                                if [ $k -gt 4 ] ; then
                                        string="$string%20AND%20$sensormode"
                                else
                                        string=$sensormode
                                fi
                        fi
			if [ "$param" == "Level" ] ; then
                                level="$value"
#                                 echo Product assigned
                                if [ $k -gt 4 ] ; then
                                        string="$string%20AND%20$level"
                                else
                                        string=$level
                                fi
                        fi
			if [ "$param" == "Product" ] ; then
                                productType="productType:$value"
#                                echo Product assigned
                                if [ $k -gt 4 ] ; then
                                        string="$string%20AND%20$productType"
                                else
                                        string=$productType
                                fi
                        fi

			if [ "$param" == "OrbitDirection" ] ; then
                                direction="orbitdirection:$value"
#                                 echo Product assigned
                                if [ $k -gt 4 ] ; then
                                        string="$string%20AND%20$direction"
                                else
                                        string=$direction
                                fi
                        fi
			if [ "$param" == "RelativeOrbit" ] ; then
				track=$value
				relativeorbitnumber="relativeorbitnumber:$value"
#				echo Relative Orbit assigned
				if [ $k -gt 4 ] ; then
                                        string="$string%20AND%20$relativeorbitnumber"
				else 	
					string=$relativeorbitnumber
                                fi
			fi
#			if [ "$param" == "FirstDay" ] ; then
#				day=$value
#				firstday="beginPosition:[NOW-10MONTHS%20TO%20NOW]"
#				if [ $k -gt 4 ] ; then
#                                        string="$string%20AND%20$firstday"
#                                else
#                                        string=$firstday
#                                fi
#			fi
			if [ "$param" == "period" ] ; then
				firstday=${value:0:10}
				lastday=${value:(-10)}
				periodsearch="(%20beginPosition:[${firstday}T00:00:00.000Z%20TO%20${lastday}T23:59:59.999Z]%20AND%20endPosition:[${firstday}T00:00:00.000Z%20TO%20${lastday}T23:59:59.999Z]%20)"
				periodflag="true"
				if [ $k -gt 4 ] ; then
                                        string="$string%20AND%20$periodsearch"
                                else
                                        string=$periodsearch
                                fi
			fi
			if [ "$param" == "polarisation" ] ; then
				pol=$value
				pol=`echo $pol  | sed 's/ /%20/'`
                                polarisation="polarisationmode:$pol"
#                                echo Polarisation Mode  assigned
				if [ $k -gt 4 ] ; then
                                        string="$string%20AND%20$polarisation"
				else
					string=$polarisation
                                fi
			fi
			if [ "$param" == "ROI" ] ; then
				ROI=$value
				if [ -e $ROI ] ; then
					poly=`cat $ROI | sed 's/, /,/g' | sed 's/ /%20/g'`
					footprint="%22Intersects(POLYGON(($poly)))%22"
					if [ $k -gt 4 ] ; then
                                  		string="$string%20AND%20footprint:$footprint"
					else
						string="footprint:$footprint"
	                                fi

					#search="https://scihub.copernicus.eu/apihub/search?q=polarisationmode:VV AND footprint:"'"Intersects(POLYGON((-4.53 29.85, 26.75 29.85, 26.75 46.80,-4.53 46.80,-4.53 29.85)))"'"
	                                
				else
					echo "ERROR: ROI file not found in running directory"
					exit 1
				fi
			fi	
			if [ "$param" == "project_functions" ] ; then
                        	if [ "$value" == "true" ] ; then
					downloadflag=$value
				fi
                        fi
			if [ "$param" == "unzip" ] ; then
                        	if [ "$value" == "true" ] ; then
					unzipflag=$value
				fi
                        fi
			if [ "$param" == "destinationfolder"  ] ; then
				if [ ! -z "$value" ] ; then
					if [ ! -d $value ] ; then
						echo "Creating folder $value" 
						mkdir -p $value
					fi
					echo "Changing directory to project_functions to : $value"
					cd $value 
				else
					cd $currentdir
					echo "Download process running in current directory"
				fi
			fi
		done
                if [ "$periodflag" == "false" ] ; then
	                firstday="(%20beginPosition:[NOW-${months}MONTHS%20TO%20NOW]%20AND%20endPosition:[NOW-${months}MONTHS%20TO%20NOW]%20)"
	                if [ $k -ge 4 ] ; then
        	                string="$string%20AND%20$firstday"
                 	else
                        	string=$firstday
                        fi
		fi
		if [ $k -ge 4 ] ; then
                	string="$string%20&rows=100000"
                fi
                echo "https://scihub.copernicus.eu/apihub/search?q=$string"
                echo "https://scihub.copernicus.eu/apihub/search?q=$string" > search

		wget --user=$username --password=$pass `cat search`  -O resultquery.html
		
	fi
# Script for splitting the original xml $1 file into $2 first files between entry
# Converting the html into xml format

echo "..............................................................."
echo "... Converting query result into individual info files........."
echo "..............................................................."

s=resultquery.html
d=`pwd`/$s
s=${s##*/}
basename=`echo ${s%.*}`
cat $d | xmllint --format - > $basename.xml
sed -i 's/&lt;/\</g' $basename.xml
sed -i 's/&gt;/\>/g' $basename.xml
cat resultquery.xml | xmllint --format - > resultquery2.xml
csplit -ksf part. resultquery2.xml /\<entry\>/ "{100000}" 2>/dev/null

# Splitting the original big file into small parts
#csplit -ksf part. resultquery.xml /\<entry\>/ "{100000}" 2>/dev/null

fi
rm -f part.00

echo "............................................................" >> metadata.log
echo "... Preparing information for sorting S1 files........." >> metadata.log
echo "............................................................" >> metadata.log

# Checking if all individual files contain a S1 file
for k in `ls part.*`
do
	test=`cat $k | grep S1`
        if [ -z "$test" ] ; then
                rm -f $k
        fi
done


# Sortening in folders
filenames=part
nfiles=`ls $filenames* | wc -l`
echo "............................................................."
echo "... There is/are "$nfiles" file/s matching with the request.." 
echo "............................................................."
if [ $nfiles -ge 1 ] ; then
	if [  "$downloadflag" == "true" ] ; then
		echo "................................................."
                echo "... Downloading process starting ................"
                echo "................................................."
		echo "................................................." >> downloading.log
		echo "... Downloading process starting ................" >> downloading.log
		echo "................................................." >> downloading.log
	fi
        for file in `ls $filenames*`
        do

		name=`cat $file | grep -o -P '(?<=\<title>).*(?=\<\/title)'`
		track=`cat $file | grep -o -P '(?<=\<int name="relativeorbitnumber">).*(?=\<\/int>)'`
		direction=`cat $file | grep -o -P '(?<=\<str name="orbitdirection">).*(?=\<\/str>)'`
                case "$direction" in
                  ASCENDING) direction="asc" ;;
                  DESCENDING) direction="dsc" ;;
                esac
                polmode=`cat $file | grep -o -P '(?<=\<str name="polarisationmode">).*(?=\<\/str)' | tr -d ' '`
                dataset=${name:4:11}
                ##store stripmap data per strip, so commented the following
                #case "$dataset" in
                #  S1A_S1_RAW_ | S1A_S2_RAW_ | S1A_S3_RAW_ | S1A_S4_RAW_ | S1A_S5_RAW_ | S1A_S6_RAW_) dataset="S1A_SM_RAW_" ;;
                #  S1A_S1_GRDF | S1A_S2_GRDF | S1A_S3_GRDF | S1A_S4_GRDF | S1A_S5_GRDF | S1A_S6_GRDF) dataset="S1A_SM_GRDF" ;;
                #  S1A_S1_GRDH | S1A_S2_GRDH | S1A_S3_GRDH | S1A_S4_GRDH | S1A_S5_GRDH | S1A_S6_GRDH) dataset="S1A_SM_GRDH" ;;
                #  S1A_S1_GRDM | S1A_S2_GRDM | S1A_S3_GRDM | S1A_S4_GRDM | S1A_S5_GRDM | S1A_S6_GRDM) dataset="S1A_SM_GRDM" ;;
                #  S1A_S1_SLC_ | S1A_S2_SLC_ | S1A_S3_SLC_ | S1A_S4_SLC_ | S1A_S5_SLC_ | S1A_S6_SLC_) dataset="S1A_SM_SLC_" ;;
                #esac
                year=${name:17:4}
                month=${name:21:2}
                day=${name:23:2}

#  Saving metadata files

		if [ ! -f ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/$name.xml ] ; then
			if [ ! -e ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day} ] ; then
				mkdir -p ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}
				cp $file ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}.xml
			else
				cp $file ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}.xml
			fi
			echo "Creating metadata file: ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}.xml" >> metadata.log
		else
                        echo "File ${name}.xml already exists. File skipped! Please check" >> metadata.log
                fi

		if [  "$downloadflag" == "true" ] ; then
			if [ ! -f ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}.zip ] && \
			[ ! -d ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name} ]; then
				echo "Downloading ${name}.zip in folder ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}" >> downloading.log
				if [ ! -d ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day} ] ; then
                                	mkdir -p ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}
					cat $file | grep "link href" | grep -o -P '(?<=href=").*(?="\/>)' > link
					wget --http-user=$username --http-password=$pass -i link -O ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}.zip
					cat $file | grep "link rel\=\"icon\"" | grep -o -P '(?<=href=").*(?="\/>)' > linkql
					wget --http-user=$username --http-password=$pass -i linkql -O ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}-ql.jpg

					if [  "$unzipflag" == "true" ] ; then
                                          unzip -qq ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}.zip \ 
						-d ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}/
                                          if [ -d ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name} ] ; then
                                            rm -rf ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}.zip
                                          fi
                                        fi

					unset value
				else
					cat $file | grep "link href" | grep -o -P '(?<=href=").*(?="\/>)' > link
					wget --http-user=$username --http-password=$pass -i link -O ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}.zip
					cat $file | grep "link rel\=\"icon\"" | grep -o -P '(?<=href=").*(?="\/>)' > linkql
					wget --http-user=$username --http-password=$pass -i linkql -O ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}-ql.jpg

					if [  "$unzipflag" == "true" ] ; then
                                          unzip -qq ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}.zip \
						-d ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}/
                                          if [ -d ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name} ] ; then
                                            rm -rf ${area}_s1_${direction}_t${track}/${dataset}_${polmode}/${year}${month}${day}/${name}.zip
                                          fi
                                        fi

					unset value
				fi
			else
				 echo "File ${name}.zip or directory ${name} already exists in the downloading folder. Image skipped! Please check" >> downloading.log
			fi
		fi
		rm -f $file
        done
fi
rm -f link
rm -f querylist* resultquery*

if [  "$downloadflag" == "true" ] ; then
	echo "................................................." >> downloading.log
	echo "... Ending of the project_functions procedure............." >> downloading.log
	echo "................................................." >> downloading.log
fi 

# Deleting intermediate files
rm -f link* filenames* part.* aux* resultq* search
cd $currentdir
