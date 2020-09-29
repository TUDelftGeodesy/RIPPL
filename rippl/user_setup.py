
print(
"""

To allow downloads for the SAR SLC's, DEMs and orbits you will need to
create a few online accounts if you want to be able to download these.

The following code will guide you through the process to setup your
working environment.

#%% md

As a first step you will have to create an account for the sentinel hub,
which can be done using the following link:

https://scihub.copernicus.eu/dhus/#/self-registration

Running the next block will ask you for your username and password and checks whether these are valid.
Sometimes the server is offline and the program will not be able to verify your account. In that case you can skip this
step as you can use the NASA server too. But be aware that you cannot use the ESA hub then! You can add your username
and password always manually to the user_settings.txt file.

"""
)

import os
import sys

sys.path.extend(['/Users/gertmulder/software/rippl_main'])

from rippl.user_settings import UserSettings
from rippl.external_dems.geoid import GeoidInterp
from rippl.external_dems.srtm.srtm_download import SrtmDownload
settings = UserSettings()

success = False
while success == False:

    print('Enter your ESA username')
    ESA_username = input()
    print('Enter your ESA password')
    ESA_password = input()

    success = settings.add_ESA_settings(ESA_username, ESA_password)

    if success:
        print('Sentinel-1 scihub account added!')

    if ESA_password == '' and ESA_username == '':
        print('Both inputs are empty, we will skip this step. You will only able to search and download data using a'
              'Earth data account')
        success = True

print(
"""

Now create an account for Earthdata which is used to download SRTM DEM data with 1 and 3 arc-second resolution
This account will also give you access to Sentinel-1 mirror server at Alaska Satellite Facility, which can be used
as an backup for the Copernicus Sentinel hub.

You can create your account using this link:
https://urs.earthdata.nasa.gov//users/new

Now go to the ASF website:
https://search.asf.alaska.edu

If you try to login it will ask you to add some things to your account and agree with the license. This makes your
account complete.

"""
)

success = False
while success == False:
    print('Enter your Earthdata username')
    NASA_username = input()
    print('Enter your Earthdata password')
    NASA_password = input()

    success = settings.add_NASA_settings(NASA_username, NASA_password)

    if success:
        print('NASA EarthExplorer account added!')

    if NASA_password == '' and NASA_username == '':
        print('Both inputs are empty, we will skip this step. You will not be able to download SRTM DEMs')
        success = True

print(
"""

Finally create an account to download data from the DLR TanDEM-X archive. This step is optional and only
needed when you work with regions above 60 degrees North or 60 degrees South of the equator, as these
areas do not have SRTM coverage. Create an account via this link:

https://sso.eoc.dlr.de/tdm90/selfservice/public/NewUser

Running the next block will check you password for the DLR website.

"""
)

success = False
while success == False:
    print('Enter your DLR username')
    DLR_username = input()
    print('Enter your DLR password')
    DLR_password = input()

    success = settings.add_DLR_settings(DLR_username, DLR_password)

    if success:
        print('DLR TanDEM-X DEM account added!')

    if DLR_password == '' and DLR_username == '':
        print('Both inputs are empty, we will skip this step. You will not be able to download TanDEM-X DEMs')
        success = True

print(
"""

Now we have the needed accounts, we can download the needed data, but you will have to define where to store the data

To do so you will need to create a folder to store:
1. A folder to store the downloaded SAR data. 
2. A folder to store the downloaded DEM data.
3. A folder to store the orbit files. These files are used to determine the exact location at satellite overpass and 
is needed to apply a correct geolocation on the ground. 
4. A folder to write the datastacks you process. 

Or:

1. Define one master folder where all other folders will be created automatically. 

Be sure that you have around 50 GB of disk space free to do the processing!

If you want you can also change the used names for the different satellite systems

"""
)

print('Do you want to change the default DEM (SRTM, TanDEM-X) or SAR sensor (Sentinel-1, TanDEM-X, Radarsat) names? '
      'This will only affect the naming not the processing itself. Type y/yes to confirm or leave empty to skip.')
change_names = input()
if change_names in ['y', 'yes']:
    print('Type your preferred naming or leave empty to use default')
    sar_sensors = []
    sar_names = []
    dem_sensors = []
    dem_names = []

    print('Name for Sentinel-1 data (default > Sentinel-1)')
    sentinel = input()
    sar_sensors.append('sentinel1')
    if input == '':
        sar_names.append('Sentinel-1')
    else:
        sar_names.append(sentinel)

    print('Name for TanDEM-X data (default > TanDEM-X)')
    tdx = input()
    sar_sensors.append('tdx')
    dem_sensors.append('tdx')
    if input == '':
        sar_names.append('TanDEM-X')
        dem_names.append('TanDEM-X')
    else:
        sar_names.append(tdx)
        dem_names.append(tdx)

    print('Name for SRTM data (default > SRTM)')
    srtm = input()
    dem_sensors.append('srtm')
    if input == '':
        dem_names.append('SRTM')
    else:
        dem_names.append(srtm)

    # If other satellite systems are added later on, these can be added below.

    settings.define_sensor_names(dem_sensors, dem_names, sar_sensors, sar_names)
else:
    settings.dem_sensors()

success = False
while success == False:
    print('Enter your main folder. Leave empty if you define other folders seperately. If used you can skip the other'
          'folders.')
    main_folder = input()
    print('Define radar database folder')
    radar_database = input()
    print('Define DEM database folder')
    DEM_database = input()
    print('Define orbit database folder')
    orbit_database = input()
    print('Define radar datastacks folder')
    radar_datastacks = input()
    print('Define NWP model database')
    NWP_model_database = input()

    success = settings.save_data_database(main_folder=main_folder,
                            radar_database=radar_database,
                            radar_datastacks=radar_datastacks,
                            orbit_database=orbit_database,
                            DEM_database=DEM_database,
                            NWP_model_database=NWP_model_database)

    if success:
        print('Folders for processing are set!')

print(
"""

Before we can download DEM data we will need to index the SRTM database and download the world geoid file. This is done
in the next step. Indexing can take a few minutes...

"""
)

# Start by downloading the geoid file
egm = GeoidInterp.create_geoid(egm_96_file=os.path.join(settings.DEM_database, 'geoid', 'egm96.dat'))

# Then index the SRTM data
filelist = SrtmDownload.srtm_listing(os.path.join(settings.DEM_database, settings.dem_sensor_name['srtm']), settings.NASA_username, settings.NASA_password)

print(
"""

To finish the setup of RIPPL install the snaphu software, which is used to do unwrapping. If you are not planning on
doing any unwrapping this is not needed.

On a linux machine this can be done by using the following command:
apt-get install snaphu (add sudo in front if needed)

On a windows machine you could use the prebuild version that is used in the STEP software from ESA. Follow this link:
http://step.esa.int/main/third-party-plugins-2/snaphu/

On a macbook you will have to build the program yourself. You can use the post under the following linke as a reference:
https://forum.step.esa.int/t/installing-snaphu-on-macbook/10969

The source code can be found under:
https://web.stanford.edu/group/radar/softwareandlinks/sw/snaphu/

Make sure that the program is in your system path!

"""
)

success = False
while success == False:
    print('Enter snaphu path')
    snaphu_path = input()

    success = settings.add_snaphu_path(snaphu_path=snaphu_path)

    if snaphu_path == '':
        print('Skipping snaphu path installation. Unwrapping will not be possible.')


# Save all settings to disk.
settings.save_settings()

print('User settings saved and validated!')
