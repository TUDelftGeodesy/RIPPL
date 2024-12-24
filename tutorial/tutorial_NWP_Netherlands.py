"""
This is a copy of the jupyter notebook tutorial, which can be run as a python script
"""

"""
This notebook gives an example of InSAR processing and the computation of a weather model delay values, that can be used for phase correction. The code to do this is included within the RIPPL package and currently uses two data sources:

1. ERA5 model and pressure level data, which can be downloaded from CDS for free. (Make sure to create an account for this)
2. CERRA model and pressure level data, which can be downloaded from CDS for free. (Make sure to create an account for this)
3. Harmonie-Arome model data, which is used in a number of european countries. In this case we make use of data provided by the royal Dutch metereological service (KNMI). Here we download two cases made available for download by them, but historical data cannot directly be downloaded from their website. This is therefore more illustrative for what someone would get from using a high-resolution model.

To do the processing, first the InSAR data is processed, an interferogram is created and unwrapped, 
using a dataset covering most of the Netherlands and Belgium. The exact steps will not be explained here as they are 
already given in the other tutorial. Please start with the Hawaii case if you are new to the software.

The main processing contains the following steps:
1. Area selection, downloading and reading in of the SAR SLCs
2. Processing to an unwrapped interferogram with an equidistant grid of 250 m
3. Downloading and processing of Harmonie data
4. Downloading and processing of ERA5 data
"""

"""
In the following code block the shape of the Benelux is loaded. Because we do not want to use external files to load the 
geometry, it is directly given in the code.
"""

# Define area of interest
import rippl
from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes

Benelux_shape = [[7.218017578125001, 53.27178347923819],
                 [5.218505859375, 53.50111704294316],
                 [4.713134765624999, 53.20603255157844],
                 [3.3508300781249996, 51.60437164681676],
                 [3.8452148437499996, 50.127621728300475],
                 [4.493408203125, 49.809631563563094],
                 [6.35009765625, 49.36806633482156],
                 [6.83349609375, 52.5897007687178],
                 [7.218017578125001, 53.27178347923819]]
study_area = ReadWriteShapes()
study_area(Benelux_shape)
shape = study_area.shape.buffer(0.2)

"""
Now the SAR data is downloaded for Sentinel-1 track 37 and the processing stack is initialized.
"""

import datetime
import numpy as np
from rippl.processing_templates.NWP_processing import NWP_Processing

# Track and data type of Sentinel data
mode = 'IW'
product_type = 'SLC'
polarisation = ['VV']

# Create the list of the 4 different stacks.
track_no = 37
stack_name = 'Benelux_NWP_track_37'
no_processes = 6

# For every track we have to select a primary date. This is based on the search results earlier.
# Choose the date with the lowest coverage to create an image with only the overlapping parts.
reference_date = datetime.datetime(year=2017, month=7, day=24)
start_date = datetime.datetime(year=2017, month=7, day=16)
end_date = datetime.datetime(year=2017, month=7, day=28)
aps_processing = NWP_Processing(processes=no_processes, stack_name=stack_name)
aps_processing.download_sentinel_data(start_date=start_date, end_date=end_date, track=track_no,
                                      shapefile=study_area.shape, data=True, source='ASF')
aps_processing.create_sentinel_stack(start_date=start_date, end_date=end_date, reference_date=reference_date,
                                    cores=no_processes, track_no=track_no, polarisation=polarisation,
                                    shapefile=study_area.shape, mode=mode, product_type=product_type)

# Finally load the stack itself. If you want to skip the download step later, run this line before other steps!
aps_processing.read_stack(start_date=start_date, end_date=end_date)
aps_processing.create_coverage_shp_kml_geojson()

"""
After downloading the SLC data and initializing the stack, we download and create the DEM, which is needed for further processing.

After creation of the DEM we also calculate the geometry for the individual radar pixels.
"""

# Download DEM
dem_buffer = 0          # Buffer around radar image where DEM data is downloaded
dem_rounding = 0        # Rounding of DEM size in degrees
min_height = -100       # Expected minimum elevation in area of interest (take geoid into account!)
max_height = 1000       # Expected maximum elevation in area of interest
dem_type = 'TDM30'      # DEM type of data we download (SRTM1, SRTM3, TDM30 and TDM90 are supported) Preferred grid is
                        # TDM30 DEM as it has the highest resolution, worldwide coverage and is the most recent.

# Resolution of output georeferenced grid
dy = 250
dx = 250
aps_processing.create_ml_coordinates(name='mercator_250m', coor_type='projection', oblique_mercator=True, dx=dx, dy=dy, buffer=dem_buffer, rounding=dem_rounding, min_height=min_height, max_height=max_height, overwrite=True)

# Define both the coordinate system of the DEM, download the needed tiles and import the DEM
aps_processing.create_dem_coordinates(dem_type=dem_type, buffer=dem_buffer, rounding=dem_rounding,
                                     min_height=min_height, max_height=max_height)
aps_processing.download_external_dem(n_processes=no_processes, dem_type=dem_type)

"""
Based on the calculated geometries we can now coregister and resample all secondary images to the reference image. 

Using the resampled image also the interferogram, coherence, amplitude and unwrapped interferogram are created and 
exported as a geotiff image. You can use GIS programs like QGIS to visualize this data.
"""

# Do the resampling and create interferograms
# Geocoding of image
aps_processing.geocode_calc_geometry(dem_type=dem_type)

# Next step applies resampling and phase correction in one step.
# Polarisation
aps_processing.coregister_resample(polarisation=polarisation)

# Create interferograms
aps_processing.create_interferogram_network(max_temporal_baseline=30)

# Resolution of output georeferenced grid
dy = 250
dx = 250
aps_processing.create_ml_coordinates(name='mercator_250m', coor_type='projection', oblique_mercator=True, dx=dx, dy=dy, buffer=dem_buffer, rounding=dem_rounding, min_height=min_height, max_height=max_height, overwrite=True)

# The creation of calibrated amplitude, interferograms, coherence and unwrapped interferograms
aps_processing.geocode_calc_geometry_multilooked(ml_name='mercator_250m', dem_type=dem_type)
aps_processing.calc_calibrated_amplitude(polarisation=polarisation, ml_name='mercator_250m')
aps_processing.calc_interferogram_coherence(polarisation=polarisation, ml_name='mercator_250m')
aps_processing.unwrap(polarisation, ml_name='mercator_250m')

# Create output geotiffs
aps_processing.create_output_geotiffs('calibrated_amplitude', ml_name='mercator_250m')
aps_processing.create_output_geotiffs('coherence', ml_name='mercator_250m')
aps_processing.create_output_geotiffs('interferogram', ml_name='mercator_250m')
aps_processing.create_output_geotiffs('unwrap', ml_name='mercator_250m')

"""
Plot the resulting values for the amplitude, coherence, interferogram and unwrapped interferogram.
"""

# Create figures
aps_processing.plot_figures(process_name='calibrated_amplitude', variable_name='calibrated_amplitude_db',
                           margins=0.1, ml_name='mercator_250m', cmap='Greys_r', overwrite=True,
                           title='Calibrated Amplitude', cbar_title='dB')
aps_processing.plot_figures(process_name='intensity', variable_name='number_of_samples', overwrite=True,
                           margins=0.1, ml_name='mercator_250m', cmap='Greys_r',
                           title='Number of samples', cbar_title='#', quantiles=[0.001, 0.999])
aps_processing.plot_figures(process_name='dem', variable_name='dem', overwrite=True,
                           margins=0.1, ml_name='mercator_250m', cmap='terrain',
                           title='DEM', cbar_title='meters')
aps_processing.plot_figures(process_name='radar_geometry', variable_name='incidence_angle', overwrite=True,
                           margins=0.1, ml_name='mercator_250m', cmap='Greys_r',
                           title='Incidence Angle', cbar_title='degrees')
aps_processing.plot_figures(process_name='coherence', variable_name='coherence', overwrite=True,
                           margins=0.1, ml_name='mercator_250m', cmap='Greys_r',
                           title='Coherence', cbar_title='coherence')
aps_processing.plot_figures(process_name='interferogram', variable_name='interferogram', overwrite=True,
                           margins=0.1, ml_name='mercator_250m', cmap='jet',
                           title='Interferogram', cbar_title='radians', remove_sea=True)
aps_processing.plot_figures(process_name='unwrap', variable_name='unwrapped', overwrite=True,
                           margins=0.1, ml_name='mercator_250m', cmap='jet',
                           title='Unwrapped interferogram', cbar_title='meter', remove_sea=True,
                           factor=-0.0554657 / (np.pi * 2) / 2,
                           dB_lims=[-18, 10], coh_lims=[0.05, 1])

"""
Now the geometry for the calculation of NWP model delays are computed. Here we use a 2.5 km grid, because the both 
weather models have a grid size larger than 2.5 km. To do a ray-tracing through the model this input geometry is needed.
"""

# Processing based on NWP model data
# Create basic geometry for APS calculations.
# Create DEM and geometry for the APS interpolation grid
dy = 2500
dx = 2500

# The creation of calibrated amplitude, interferograms, coherence and unwrapped interferograms
aps_processing.create_ml_coordinates(name='mercator_2500m', coor_type='projection', oblique_mercator=True, dx=dx, dy=dy, 
                                     buffer=dem_buffer, rounding=dem_rounding,
                                     min_height=min_height, max_height=max_height, overwrite=True)


aps_processing.geocode_calc_geometry_multilooked(ml_name='mercator_2500m', dem_type=dem_type, incidence_angle=True,
                                                 azimuth_angle=True)
aps_processing.geocode_calc_geometry_multilooked(ml_name='mercator_250m', dem_type=dem_type, incidence_angle=True,
                                                 azimuth_angle=True)

"""
In the following section we download the needed ECMWF data. For this analysis two different reanalysis datasets are downloaded:
- The ERA5 renanalysis, which is a worldwide reanalysis on a ~30 km grid
- The CERRA reanalysis, which is a european reanalysis on a 5.5 km grid

Although both datasets are available on model and pressure levels, we prefer to use the pressure level data as it is 
smaller and much faster to download. However, users can also download the model level data, which should give a small 
increase in accuracy of the delay estimates.
"""

# Download needed data
from rippl.NWP_model_delay.load_NWP_data.ecmwf.ecmwf_download import CDSdownload
from rippl.user_settings import UserSettings

settings = UserSettings()
settings.load_settings()
# Download of ERA5 data. (This can take some time because you have to wait in the queue!)
# Make sure you created your own cdsapi token!
# Find the guidelines for downloading via https://cds.climate.copernicus.eu/api-how-to

# We project_functions data for western europe. (This can be extended or narrowed, but should at least include the satellite
# orbit and radar points on the ground.)
latlim = [45, 56]
lonlim = [-2, 12]

data_types = ['reanalysis-era5-pressure-levels', 'reanalysis-cerra-pressure-levels']
time_interp = 'nearest'

# Take only part of dataset
overpasses = aps_processing.get_overpasses()

for data_type in data_types:
    print('Downloading ' + data_type)
    download_aps = CDSdownload(overpass_times=overpasses, latlim=latlim, lonlim=lonlim, data_type=data_type, processes=1)
    download_aps.prepare_download()
    download_aps.download()

"""
Now the atmospheric delays are calculated. This is done using a ray-tracing technique where the path through the NWP 
model grid is calculated. Then, the temperature, pressure and humidity values along the ray's path are used to calculate the delay. 

To compare the results with the unwrapped delay values from InSAR data, a synthetic interferogram is created from the 
delay values of two atmospheric states. Results are written to disk as geotiff and .png files.
"""

# Create image for ERA5 and CERRA

# The first step we take is calculate the images for the two dates
# Then do the APS calculations
for model_name in ['cerra', 'era5']:
    aps_processing.calculate_aps(ml_name_ray_tracing='mercator_2500m', ml_name='mercator_250m', model_name=model_name,
                                 model_level_type='pressure_levels', latlim=latlim, lonlim=lonlim,
                                 time_correction=False, geometry_correction=True, spline_type='linear')

for model_name in ['cerra', 'era5']:
    # Then we create the interferogram of both images
    aps_processing.calculate_ifg_aps(ml_name='mercator_250m', model_name=model_name,
                                 latlim=latlim, lonlim=lonlim, geometry_correction=True)

aps_processing.create_output_geotiffs('cerra_nwp_delay', ml_name='mercator_250m')
aps_processing.create_output_geotiffs('cerra_nwp_interferogram', ml_name='mercator_250m')

aps_processing.create_output_geotiffs('era5_nwp_delay', ml_name='mercator_250m')
aps_processing.create_output_geotiffs('era5_nwp_interferogram', ml_name='mercator_250m')

# Create images and geotiffs for ECMWF data
for model_name in ['era5', 'cerra']:
    for data_type, name_type in zip(['', '_ifg'], ['delay', 'interferogram']):
        for delay_type in ['_aps', '_wet_delay']:
            aps_processing.plot_figures(process_name= model_name + '_nwp_' + name_type, variable_name= model_name + data_type + delay_type,
                                       margins=0.1, ml_name='mercator_250m', cmap='jet', overwrite=True,
                                       title='NWP ' + name_type + ' ' + model_name, cbar_title='meter', remove_sea=True)

"""
The following steps will use Harmonie data that was provided by the KNMI (knmi.nl), the Royal Dutch Meteorological Institute
This data is added to the paper to run this tutorial, but is not directly downloadable from their databases. Data is
available upon request with a processing fee. For other national forecast agencies within europe other rules could apply.
"""

import os
from rippl.user_settings import UserSettings
import urllib.request

settings = UserSettings()
settings.load_settings()

# Download of Harmonie data. This data is stored with the paper itself (should go relatively fast)
urls = ['https://surfdrive.surf.nl/files/index.php/s/kwXjZoRqujHRjsC/download',
        'https://surfdrive.surf.nl/files/index.php/s/N8RCpbFZjPrY1ME/download']
filenames = ['HA38_N25_201707180300_00245_GB',
             'HA38_N25_201707240300_00245_GB']
download_folder = os.path.join(settings.settings['paths']['NWP_model_database'], 'harmonie', 'h38')
if not os.path.exists(download_folder):
    os.mkdir(download_folder)

for url, filename in zip(urls, filenames):
    if not os.path.exists(os.path.join(download_folder, filename)):
        urllib.request.urlretrieve(url, os.path.join(download_folder, filename))

"""
Using ray-tracing also the expected InSAR delays from weather model data for the Harmonie model are computed.
"""

# The first step we take is calculate the images for the two dates
# Then do the APS calculations
aps_processing.calculate_aps(ml_name_ray_tracing='mercator_2500m', ml_name='mercator_250m', model_name='harmonie',
                             model_level_type='model_levels', time_correction=False, geometry_correction=True,
                             spline_type='linear')

# Then we create the interferogram
aps_processing.calculate_ifg_aps(ml_name='mercator_250m', model_name='harmonie', geometry_correction=True)

# Create image and .tiff file for ERA5 individual dates and ifg
aps_processing.create_output_geotiffs('harmonie_nwp_delay', ml_name='mercator_250m')
aps_processing.create_output_geotiffs('harmonie_nwp_interferogram', ml_name='mercator_250m')

# Create images and geotiffs for ECMWF data
for data_type, name_type in zip(['', '_ifg'], ['delay', 'interferogram']):
    for delay_type in ['_aps', '_wet_delay']:
        aps_processing.plot_figures(process_name='nwp_' + name_type, variable_name= 'harmonie' + data_type + delay_type,
                                   margins=0.1, ml_name='mercator_250m', cmap='jet',
                                   title='NWP ' + name_type + ' ' + 'harmonie', cbar_title='radians', remove_sea=True)
