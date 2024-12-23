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
aps_processing.download_sentinel_data(start_date=start_date, end_date=end_date, track=track_no,                                                                             shapefile=study_area.shape, data=True, source='ASF')
aps_processing.create_sentinel_stack(start_date=start_date, end_date=end_date, reference_date=reference_date,
                                    cores=no_processes, track_no=track_no, polarisation=polarisation,
                                    shapefile=study_area.shape, mode=mode, product_type=product_type)

# Finally load the stack itself. If you want to skip the download step later, run this line before other steps!
aps_processing.read_stack(start_date=start_date, end_date=end_date)
aps_processing.create_coverage_shp_kml_geojson()


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

# Create figures
aps_processing.plot_figures(process_name='calibrated_amplitude', variable_name='calibrated_amplitude_db',
                           margins=-0.25, ml_name='mercator_250m', cmap='Greys_r',
                           title='Calibrated Amplitude', cbar_title='dB')
aps_processing.plot_figures(process_name='intensity', variable_name='number_of_samples',
                           margins=-0.25, ml_name='mercator_250m', cmap='Greys_r',
                           title='Number of samples', cbar_title='#', quantiles=[0.001, 0.999])
aps_processing.plot_figures(process_name='dem', variable_name='dem',
                           margins=-0.25, ml_name='mercator_250m', cmap='terrain',
                           title='DEM', cbar_title='meters')
aps_processing.plot_figures(process_name='radar_geometry', variable_name='incidence_angle',
                           margins=-0.25, ml_name='mercator_250m', cmap='Greys_r',
                           title='Incidence Angle', cbar_title='degrees')
aps_processing.plot_figures(process_name='coherence', variable_name='coherence',
                           margins=-0.25, ml_name='mercator_250m', cmap='Greys_r',
                           title='Coherence', cbar_title='coherence')
aps_processing.plot_figures(process_name='interferogram', variable_name='interferogram',
                           margins=-0.25, ml_name='mercator_250m', cmap='jet',
                           title='Interferogram', cbar_title='radians', remove_sea=True)
aps_processing.plot_figures(process_name='unwrap', variable_name='unwrapped',
                           margins=-0.25, ml_name='mercator_250m', cmap='jet',
                           title='Unwrapped interferogram', cbar_title='meter', remove_sea=True,
                           factor=-0.0554657 / (np.pi * 2) / 2,
                           dB_lims=[-18, 10], coh_lims=[0.05, 1])

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
            aps_processing.plot_figures(process_name='nwp_' + name_type, variable_name= model_name + data_type + delay_type,
                                       margins=-0.25, ml_name='mercator_250m', cmap='jet',
                                       title='NWP ' + name_type + ' ' + model_name, cbar_title='radians', remove_sea=True)
