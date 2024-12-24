"""
This is a copy of the jupyter notebook with the same name.
"""

"""
In this notebook we will explain the RIPPL workflow based on a study case about the earthquake on Hawaii on May 4th 2018.

First, you have to insert the path to your local RIPPL package.

Next, you can define the study area. There are different options to do so.
1. Create a shapefile (this can be done using ArcGIS or QGIS software)
2. Create a kml file using google earth > https://www.google.com/earth/
3. Create a geojson using > http://geojson.io
4. Create a list of coordinates in lat/lon coordinate pairs. This is what we will do here. 

To show the study area it is possible to run both google earth and geojson within the notebook.

For further background on this event you can read:
https://en.wikipedia.org/wiki/2018_Hawaii_earthquake

Following image shows the earthquake strength (credits: USGS 2018):
![Image of Hawaii earthquake strength](https://upload.wikimedia.org/wikipedia/commons/2/2d/2018_Hawaii_earthquake.jpg)
"""

import numpy as np
import datetime

from rippl.SAR_sensors.sentinel.sentinel_image_download import DownloadSentinel
from rippl.orbit_geometry.read_write_shapes import ReadWriteShapes

Hawaii_shape = [(-155.75, 18.90), (-155.75, 20.2), (-154.75, 19.50), (-155.75, 18.90)]
study_area = ReadWriteShapes()
study_area(Hawaii_shape)

geojson = study_area.shape

"""
The next step in the processing is selection of the right track, as the Sentinel-1 satellite will cover the area on both
ascending and descending tracks. However, to do so, we will have to define the start and end date of our data search,
because the satellite is not always acquiring data.
Next search will give a small oversight of the available tracks during our period of interest. In our case we will
search one week before and one week after the earthquake.

Watch out with extending the search window, this can result in a lot of images. Note that every downloaded file will take approximately 4GB of disk space!

For the selected period of interest there are four different options:
1. Use one date and a time window around that date in days > date=date1, time_window=x
2. Use multiple dates and a time window around those dates in days > date=[date1, date2], time_window=x
3. Use a start and end date > start_date=date1, end_date=date2
4. Use multiple start and end dates > start_dates=[date1, date3], end_dates=[date2, date4]

In this case we use the date of the earthquak itself and a 12-day time window.
"""

# Track and data type of Sentinel data
mode = 'IW'
product_type = 'SLC'
polarisation = ['VV','VH']

# First we check using a time window
earthquake_date = datetime.datetime(year=2018, month=5, day=4, hour=22)
time_window = datetime.timedelta(days=12)

find_track = DownloadSentinel(date=earthquake_date, time_window=time_window,
                              shape=study_area.shape, sensor_mode=mode,
                              polarisation=polarisation)

find_track.sentinel_search_ASF()
find_track.summarize_search_results(plot_cartopy=True, buffer=2)

"""
After selection of the desired track we can start the actual download of the images. In our case we use track 087.

This will download our data automatically to our radar database. Additionally, it will download the precise orbit files.
These files are created within a few weeks after the data acquisition and define the satellite orbit within a few cm
accuracy. These orbits are necessary to accurately define the positions of the radar pixels on the ground later on
in the processing.
"""

from rippl.processing_templates.InSAR_processing import InSAR_Processing

# Create the list of the 4 different stacks.
track_no = 87
stack_name = 'Hawaii_may_2018_descending'
# For every track we have to select a master date. This is based on the search results earlier.
# Choose the date with the lowest coverage to create an image with only the overlapping parts.
start_date = datetime.datetime(year=2018, month=4, day=22)
end_date = datetime.datetime(year=2018, month=5, day=8)
reference_date = datetime.datetime(year=2018, month=5, day=5)

# Number of processes for parallel processing. Make sure that for every process at least 2GB of RAM is available
no_processes = 6
s1_processing = InSAR_Processing(processes=no_processes, stack_name=stack_name)
s1_processing.download_sentinel_data(start_date=start_date, end_date=end_date, track=track_no,
                                     polarisation=polarisation, shapefile=study_area.shape, data=True, source='ASF')
s1_processing.create_sentinel_stack(start_date=start_date, end_date=end_date, reference_date=reference_date,
                                    cores=no_processes, track_no=track_no, polarisation=polarisation,
                                    shapefile=study_area.shape, mode=mode, product_type=product_type)

# Finally load the stack itself. If you want to skip the download step later, run this line before other steps!
s1_processing.read_stack(start_date=start_date, end_date=end_date)
s1_processing.create_coverage_shp_kml_geojson()

"""
To define the location of the radar pixels on the ground we need the terrain elevation. Although it is possible to 
derive terrain elevation from InSAR data, our used Sentinel-1 dataset is not suitable for this purpose. Therefore, we
download data from an external source to create a digital elevation model (DEM). In our case we use SRTM data. 

However, to find the elevation of the SAR data grid, we have to resample the data to the radar grid first to make it
usable. This is done in the next steps.
"""

# Some basic settings for DEM creation.
dem_buffer = 0          # Buffer around radar image where DEM data is downloaded
dem_rounding = 0        # Rounding of DEM size in degrees
min_height = -100       # Expected minimum elevation in area of interest (take geoid into account!)
max_height = 4300       # Expected maximum elevation in area of interest
dem_type = 'SRTM1'      # DEM type of data we download (SRTM1, SRTM3 and TanDEM-X are supported)

# Define both the coordinate system of the DEM, download the needed tiles and import the DEM
s1_processing.create_dem_coordinates(dem_type=dem_type, buffer=dem_buffer, rounding=dem_rounding,
                                     min_height=min_height, max_height=max_height)
s1_processing.download_external_dem(n_processes=no_processes, dem_type=dem_type)

"""
Using the obtained elevation model the exact location of the radar pixels in cartesian (X,Y,Z) and geographic (Lat/Lon)
can be derived. This is only done for the reference SLC. This process is referred to as geocoding.
"""

# Geocoding of image
s1_processing.geocode_calc_geometry(dem_type=dem_type)

"""
The information from the geocoding can directly be used to find the location of the reference grid pixels in the secondary
grid images. This process is called coregistration. Because the orbits are not exactly the same with every satellite 
overpass but differ hundreds to a few thousand meters every overpass, the grids are slightly shifted with respect to 
each other. These shift are referred to as the spatial baseline of the images. To correctly overlay the reference and secondary
images the software coregisters and resamples to the reference grid.

To do so the following steps are done:
1. Coregistration of secondary to reference image
2. Deramping the doppler effects due to TOPs mode of Sentinel-1 satellite
3. Resampling of secondary image
4. Reramping resampled secondary image.

Due to the different orbits of the reference and secondary image, the phase of the radar signal is also shifted. We do not 
know the exact shift of the two image, but using the geometry of the two images we can estimate the shift of the phase
between different pixels. Often this shift is split in two contributions:
1. The flat earth phase. This phase is the shift in the case the earth was a perfect ellipsoid
2. The topographic phase. This is the phase shift due to the topography on the ground.
In our processing these two corrections are done in one go.
"""

# Next step applies resampling and phase correction in one step.
# Polarisation
s1_processing.coregister_resample(polarisation=polarisation)

"""
Before we create an interferogram the different bursts are first mosaicked. This can only be done after resampling as
it is influenced by the phase ramps in TOPs mode of Sentinel-1. 

The independent SAR grids can now be visualized using the amplitude of the resampled data. In our case these are 
written as .tiff files for a georeferenced grid of the region. The data can be visualized using QGIS or other GIS software. The amplitude power
is given in dB. 
"""

# Resolution of output georeferenced grid
dlat = 0.001
dlon = 0.001

# The actual creation of the calibrated amplitude images
s1_processing.create_ml_coordinates(name='geographic_100m', coor_type='geographic', dlat=dlat, dlon=dlon, buffer=dem_buffer, rounding=dem_rounding, min_height=min_height, max_height=max_height)
# To calculate the calibrated amplitudes we need the incidence angle.
s1_processing.geocode_calc_geometry_multilooked(ml_name='geographic_100m')
s1_processing.calc_calibrated_amplitude(polarisation=polarisation, ml_name='geographic_100m')

# Create the output tiffs
s1_processing.create_output_geotiffs('calibrated_amplitude', ml_name='geographic_100m')

"""
We can do the same thing using a projected grid with distances in meters. In this case we use the oblique mercator
projection. This projection can be configured in such a way that the grid follows the orbit of the satellite, which has
the advantage that the final product will be smaller, has less empty spaces and respects the satellite azimuth and
range directions.
"""

# Resolution of output georeferenced grid
dy = 100
dx = 100

# The actual creation of the calibrated amplitude images
s1_processing.create_ml_coordinates(name='mercator_100m', coor_type='projection', oblique_mercator=True, dx=dx, dy=dy, buffer=dem_buffer, rounding=dem_rounding, min_height=min_height, max_height=max_height, overwrite=True)
# To calculate the calibrated amplitudes we need the incidence angle.
s1_processing.geocode_calc_geometry_multilooked(ml_name='mercator_100m')
s1_processing.calc_calibrated_amplitude(polarisation=polarisation, ml_name='mercator_100m')

# Create the output geotiff files
s1_processing.create_output_geotiffs('calibrated_amplitude', ml_name='mercator_100m')

"""
After moasicing we can create the interferogram between the different images. This image is also multilooked and 
outputted as a .tiff file. This can also be viewed using GIS software. Because the phase shift between different pixels is often 
larger than two pi radians or a wavelength (56 mm for C-band), this image will show fringes going from -pi to pi and 
starting at -pi again. 
Using the same multilooking grid also a coherence grid is created, which indicates the quality of the obtained phases.
"""

# Create interferograms meta data
s1_processing.create_interferogram_network(max_temporal_baseline=30)

# Resolution of output georeferenced grid
dy = 100
dx = 100

# The actual creation of the calibrated amplitude images
s1_processing.create_ml_coordinates(name='mercator_100m', coor_type='projection', oblique_mercator=True, dx=dx, dy=dy, buffer=dem_buffer, rounding=dem_rounding, min_height=min_height, max_height=max_height, overwrite=True)

# Create interferograms and coherence
s1_processing.calc_interferogram_coherence(polarisation=polarisation, ml_name='mercator_100m')
s1_processing.calc_interferogram_coherence(polarisation=polarisation, ml_name='geographic_100m')

# Create output geotiffs
s1_processing.create_output_geotiffs('coherence', ml_name='mercator_100m')
s1_processing.create_output_geotiffs('interferogram', ml_name='mercator_100m')
s1_processing.create_output_geotiffs('coherence', ml_name='geographic_100m')
s1_processing.create_output_geotiffs('interferogram', ml_name='geographic_100m')

"""
To go to absolute differences the data is therefore unwrapped. The result of this is given in the unwrapped geotiff.
For the unwrapping we use the program snaphu. With a resolution of 100 meters the unwrapping can take quite some time.
If you want to speed up the unwrapping you could go for larger grid cells.
"""

# To do unwrapping we use the program Snaphu. For high resolution images this can take a very long time.
# Therefore, we change the resolution, to create a higher number of looks and more stable
# interferometric phase signal.
dy = 200
dx = 200

# The actual creation of the calibrated amplitude images
s1_processing.create_ml_coordinates(name='mercator_200m', coor_type='projection', oblique_mercator=True, dx=dx, dy=dy, buffer=dem_buffer, rounding=dem_rounding, min_height=min_height, max_height=max_height, overwrite=True)
s1_processing.geocode_calc_geometry_multilooked(ml_name='mercator_200m')
s1_processing.calc_calibrated_amplitude(polarisation=polarisation, ml_name='mercator_200m')
s1_processing.calc_interferogram_coherence(polarisation=polarisation, ml_name='mercator_200m')

s1_processing.unwrap(polarisation, ml_name='mercator_200m')
s1_processing.create_output_geotiffs('unwrap', ml_name='mercator_200m')

"""

"""

# Create some images of the ifg / no looks / incidence angles / unwrapped image / coherence / cal
# amplitude

# Create images using cartopy
s1_processing.plot_figures(process_name='calibrated_amplitude', variable_name='calibrated_amplitude_db',
                           margins=-0.25, ml_name='mercator_200m', cmap='Greys_r',
                           title='Calibrated Amplitude', cbar_title='dB')
s1_processing.plot_figures(process_name='intensity', variable_name='number_of_samples',
                           margins=-0.25, ml_name='mercator_200m', cmap='Greys_r',
                           title='Number of samples', cbar_title='#', quantiles=[0.001, 0.999])
s1_processing.plot_figures(process_name='dem', variable_name='dem',
                           margins=-0.25, ml_name='mercator_100m', cmap='terrain',
                           title='DEM', cbar_title='meters')
s1_processing.plot_figures(process_name='radar_geometry', variable_name='incidence_angle',
                           margins=-0.25, ml_name='mercator_100m', cmap='Greys_r',
                           title='Incidence Angle', cbar_title='degrees')
s1_processing.plot_figures(process_name='coherence', variable_name='coherence',
                           margins=-0.25, ml_name='mercator_100m', cmap='Greys_r',
                           title='Coherence', cbar_title='coherence')
s1_processing.plot_figures(process_name='interferogram', variable_name='interferogram',
                           margins=-0.25, ml_name='mercator_100m', cmap='jet',
                           title='Interferogram', cbar_title='radians', remove_sea=True)
s1_processing.plot_figures(process_name='interferogram', variable_name='interferogram',
                           margins=-0.25, ml_name='mercator_100m', cmap='jet',
                           title='Interferogram', cbar_title='radians', remove_sea=True)
s1_processing.plot_figures(process_name='unwrap', variable_name='unwrapped',
                           margins=-0.25, ml_name='mercator_200m', cmap='jet',
                           title='Unwrapped interferogram', cbar_title='meter', remove_sea=True,
                           factor=-0.0554657 / (np.pi * 2) / 2, linear_transparency=False,
                           dB_lims=[-18, 10], coh_lims=[0.05, 1])

"""
This finishes the tutorial! In QGIS you can visualize the results of your processing.
"""