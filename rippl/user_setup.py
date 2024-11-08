import os
import sys
import logging
logging.getLogger().setLevel(logging.INFO)

logging.info(
"""

To allow downloads for the SAR SLC's, DEMs and orbits you will need to
create a few online accounts if you want to be able to download these.

The following code will guide you through the process to setup your
working environment.
""")

success = False
while success == False:

    logging.info('Enter the path to your RIPPL folder')
    rippl_path = input()

    try:
        sys.path.extend([rippl_path])
        import rippl

        success = True
    except Exception as e:
        logging.warning('Wrong RIPPL path, package cannot be imported. Try again: ' + str(e))

logging.info(
"""

As a first step you will have to create an account for the sentinel hub,
which can be done using the following link:

https://dataspace.copernicus.eu/

Running the next block will ask you for your username and password and checks whether these are valid.
Sometimes the server is offline and the program will not be able to verify your account. In that case you can skip this
step as you can use the EarthData server too. But be aware that you cannot use the Copernics hub then! You can add your username
and password always manually to the user_settings.json file.

"""
)

from rippl.user_settings import UserSettings
from rippl.external_dems.geoid import GeoidInterp
settings = UserSettings()

success = False
while success == False:

    logging.info('Enter your Copernicus username')
    copernicus_username = input()
    logging.info('Enter your Copernicus password')
    copernicus_password = input()

    success = settings.add_copernicus_settings(copernicus_username, copernicus_password)

    if success:
        logging.info('Copernicus account added!')

    if copernicus_password == '' and copernicus_username == '':
        logging.info('Both inputs are empty, we will skip this step. You will only able to search and download data using a'
              'Earth data account')
        success = True

logging.info(
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
    logging.info('Enter your EarthData username')
    earthdata_username = input()
    logging.info('Enter your EarthData password')
    earthdata_password = input()

    success = settings.add_earthdata_settings(earthdata_username, earthdata_password)

    if success:
        logging.info('EarthData account added!')

    if earthdata_password == '' and earthdata_username == '':
        logging.info('Both inputs are empty, we will skip this step. You will not be able to download SRTM DEMs')
        success = True

logging.info(
"""

Create an account to download data from the DLR TanDEM-X archive. This step is optional and only
needed when you work with regions above 60 degrees North or 60 degrees South of the equator, as these
areas do not have SRTM coverage. 

There are two different datasets that also require two different accounts. 

For the 30 meter DEM (TDM30) create an account via this link:
https://sso.eoc.dlr.de/tdm30-edited/selfservice/public/newuser
For the 90 meter DEM (TDM90) create an account via this link:
https://sso.eoc.dlr.de/tdm90/selfservice/public/newuser

Running the next block will check you password for the DLR website.

"""
)

success = False
while success == False:
    logging.info('Do you want to use DLR DEM data? (default is no)')
    use_dlr = input()

    if use_dlr.lower() == 'yes' or use_dlr.lower == 'y':
        for dem_type in ['TDM30', 'TDM90']:
            logging.info('Username EOC Geoservice (DLR) TanDEM-X DEM account for' + dem_type + ':')
            DLR_username = input()
            logging.info('Username EOC Geoservice (DLR) TanDEM-X DEM account for' + dem_type + ':')
            DLR_password = input()

            success = settings.add_DLR_settings(DLR_username, DLR_password, dem_type)

            if success:
                logging.info('EOC Geoservice (DLR) TanDEM-X DEM account for ' + dem_type + ' added!')

            if DLR_password == '' and DLR_username == '':
                logging.info('Both inputs are empty, we will skip this step. You will not be able to download TanDEM-X DEMs')
                success = True
    else:
        logging.info('Skipping DLR DEM settings')
        success = True

logging.info(
"""

After the dowload options for SAR data and DEMs we also add two options to add accounts to download NWP model data. One of them is from ERA5 (worldwide) and the other from the KNMI (Netherlands or Western Europe only). You can create accounts via the following links:

ERA5: https://cds.climate.copernicus.eu/user/register
The requested api key can then be found at:
https://cds.climate.copernicus.eu/ and clicking on your account name in the upper right corner. The UID and API key then given at the bottom of the page.
Downloads will be done using the cds api. To install the python package check out:
https://cds.climate.copernicus.eu/api-how-to

KNMI: https://developer.dataplatform.knmi.nl/register/
The API key can then be found under:
https://developer.dataplatform.knmi.nl/member/

"""
)

success = False
while success == False:
    logging.info('Do you want to use ERA5 NWP data? (default is no)')
    use_ecmwf = input()

    if use_ecmwf.lower() == 'yes' or use_ecmwf.lower == 'y':
        logging.info('API Key for Climate Data Store:')
        api_key = input()
        success = settings.add_ecmwf_settings(api_key)

        if success:
            logging.info('Climate Data Store account added!')
            success = True
    else:
        logging.info('Skipping Climate Data Store settings')
        success = True

success = False
while success == False:
    use_knmi = input('Do you want to use KNMI NWP data? (default is no)')

    if use_knmi.lower() == 'yes' or use_knmi.lower == 'y':
        logging.info('API Key for Climate Data Store:')
        api_key = input()
        success = settings.add_knmi_settings(api_key)

        if success:
            logging.info('KNMI account added!')
            success = True
    else:
        logging.info('Skipping KNMI settings')
        success = True

logging.info(
"""

Now we have the needed accounts, we can download the needed data, but you will have to define where to store the data

To do so you will need to create a folder to store:
1. A folder to store the downloaded SAR data. 
2. A folder to store the downloaded DEM data.
3. A folder to store the orbit files. These files are used to determine the exact location at satellite overpass and 
is needed to apply a correct geolocation on the ground. 
4. A folder to write the data_stacks you process. 

Or:

1. Define one primary folder where all other folders will be created automatically. 

Be sure that you have around 50 GB of disk space free to do the processing!

If you want you can also change the used names for the different satellite systems

"""
)

logging.info('Do you want to change the default DEM (SRTM, TanDEM-X) or SAR sensor (Sentinel-1, TanDEM-X, Radarsat) names? '
      'This will only affect the naming not the processing itself. Type y/yes to confirm or leave empty to skip.')
change_names = input()
if change_names in ['y', 'yes']:
    logging.info('Type your preferred naming or leave empty to use default')
    sar_names = []
    dem_names = []
    nwp_names = []

    logging.info('Name for Sentinel-1 data (default > sentinel1)')
    sentinel = input()
    sar_names.append(sentinel) if sentinel else sar_names.append('sentinel1')

    logging.info('Name for TerraSAR-X data (default > tsx)')
    tsx = input()
    sar_names.append(tsx) if tsx  else sar_names.append('tsx')

    logging.info('Name for TanDEM-X data (default > tdx)')
    tdx = input()
    dem_names.append(tdx) if tdx  else sar_names.append('tdx')

    logging.info('Name for SRTM data (default > srtm)')
    srtm = input()
    dem_names.append(srtm) if srtm else sar_names.append('srtm')

    logging.info('Name for ECWMF NWP model data (default > ecmwf)')
    ecmwf = input()
    nwp_names.append(ecmwf) if ecmwf  else sar_names.append('ecmwf')

    logging.info('Name for Harmonie-Arome NWP model data (default > harmonie)')
    harm = input()
    nwp_names.append(srtm) if harm else sar_names.append('harmonie')

    settings.define_sensor_names(dem_names, sar_names, nwp_names)
else:
    settings.define_sensor_names()

success = False
while success == False:
    logging.info('Enter your main folder. Leave empty if you define other folders seperately. If used you can skip the other'
          ' folders.')
    main_folder = input()
    logging.info('Define radar database folder')
    radar_database = input()
    logging.info('Define DEM database folder')
    DEM_database = input()
    logging.info('Define radar data products folder')
    radar_data_products = input()
    logging.info('Define orbit database folder')
    orbit_database = input()
    logging.info('Define radar data stacks folder')
    radar_data_stacks = input()
    logging.info('Define NWP model database')
    NWP_model_database = input()

    success = settings.save_data_database(main_folder=main_folder,
                            radar_database=radar_database,
                            radar_data_stacks=radar_data_stacks,
                            orbit_database=orbit_database,
                            DEM_database=DEM_database,
                            NWP_model_database=NWP_model_database,
                            radar_data_products=radar_data_products)

    if success:
        logging.info('Folders for processing are set!')

logging.info(
"""

Before we can download DEM data we will need to download the EGM96 geoid model. 
This is done in the next step.

"""
)

# Start by downloading the geoid file
egm = GeoidInterp.create_geoid(egm_96_file=os.path.join(settings.settings['paths']['DEM_database'], 'geoid', 'egm96.dat'))

logging.info(
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
    logging.info('Enter snaphu path')
    snaphu_path = input()

    success = settings.add_snaphu_path(snaphu_path=snaphu_path)

    if snaphu_path == '':
        logging.info('Skipping snaphu path installation. Unwrapping will not be possible.')
        success = True

# Save all settings to disk.
settings.save_settings()

logging.info('User settings saved and validated!')
