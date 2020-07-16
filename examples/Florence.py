from rippl.processing_templates.general_sentinel_1 import GeneralPipelines

# Settings where the data is stored
shapefile = 'AOI_Florence.kml'
stack_name = 'Florence'

# Track and data type of Sentinel data
track = 95
mode = 'IW'
product_type = 'SLC'
polarisation = ['VV']       # Possibly add HV later on

# Start, master and end date of processing
start_date = '2016-03-01'
end_date = '2016-07-01'
master_date = '2016-05-10'

# DEM type
dem_type = 'SRTM3'
dem_buffer = 1
dem_rounding = 1
lon_resolution = 6

# Multilooking coordinates
dlat = 0.0005
dlon = 0.0005

processing = GeneralPipelines(processes=8)

create_stack = False
load_stack = True
coreg = True
multilook = True
geotiff = True
plot = True

# Download and create the dataset
if create_stack:
    processing.download_sentinel_data(start_date=start_date, end_date=end_date, track=track,
                                               polarisation=polarisation, shapefile=shapefile, data=True)
    processing.create_sentinel_stack(start_date=start_date, end_date=end_date, master_date=master_date,
                                              track=track, stack_name=stack_name, polarisation=polarisation,
                                              shapefile=shapefile, mode=mode, product_type=product_type)

# Load stack
if load_stack:
    processing.read_stack(start_date=start_date, end_date=end_date, stack_name=stack_name)
    processing.create_ifg_network(network_type='temp_baseline', temporal_baseline=200)

    # Coordinate systems
    processing.create_radar_coordinates()
    processing.create_dem_coordinates(dem_type=dem_type)
    processing.create_ml_coordinates(coor_type='geographic', dlat=dlat, dlon=dlon)

# Data processing
if coreg:
    processing.download_external_dem(dem_type=dem_type, lon_resolution=lon_resolution, buffer=dem_buffer + 1, rounding=dem_rounding)
    processing.geocoding(dem_type=dem_type, dem_buffer=dem_buffer, dem_rounding=dem_rounding)
    processing.geometric_coregistration_resampling(polarisation)

# Multilooking
if multilook:
    processing.prepare_multilooking_grid(polarisation[0])
    processing.create_calibrated_amplitude_multilooked(polarisation)
    processing.create_interferogram_multilooked(polarisation)
    processing.create_coherence_multilooked(polarisation)
    processing.create_unwrapped_images(polarisation)

    # Calculate geometry
    processing.create_geometry_mulitlooked(dem_type=dem_type, dem_buffer=dem_buffer, dem_rounding=dem_rounding)

# Create the geotiffs
if geotiff:
    processing.create_output_tiffs_amplitude()
    processing.create_output_tiffs_coherence_ifg()
    processing.create_output_tiffs_geometry()
    processing.create_output_tiffs_unwrap()

if plot:
    processing.create_plots_amplitude(False)
    processing.create_plots_coherence(False)
    processing.create_plots_ifg(False)

# Finally create a stack of the output resampled SLCs for a certain region.

latlim = [43.75, 43.78]
lonlim = [11.24, 11.27]

# Get the SLC data
slc_dat = processing.stack.stack_data_iterator(['correct_phases'], coordinates=[processing.radar_coor],
                               process_types=['phase_corrected'])
slc_dates = slc_dat[1]
slcs = [slc.disk['data'] for slc in slc_dat[-1]]

# Get the deramp data
ramp_dat = processing.stack.stack_data_iterator(['calc_reramp'], coordinates=[processing.radar_coor],
                               process_types=['ramp'])
ramp_dates = slc_dat[1]
ramps = [ramp.disk['data'] for ramp in ramp_dat[-1]]

# Define the lat/lon region and get lat/lon/elevation angle/azimuth angle
lat = processing.stack.stack_data_iterator(['geocode'], coordinates=[processing.radar_coor],
                               process_types=['lat'])[-1][0].disk['data']
lon = processing.stack.stack_data_iterator(['geocode'], coordinates=[processing.radar_coor],
                               process_types=['lon'])[-1][0].disk['data']
incidence_angle = processing.stack.stack_data_iterator(['radar_ray_angles'], coordinates=[processing.radar_coor],
                               process_types=['incidence_angle'])[-1][0].disk['data']
dem = processing.stack.stack_data_iterator(['dem'], coordinates=[processing.radar_coor],
                               process_types=['dem'])[-1][0].disk['data']

import numpy as np
import os
from rippl.meta_data.image_data import ImageData

# Get extend
valid_lines = (np.max(lat, axis=1) > latlim[0]) * (np.min(lat, axis=1) < latlim[1]) * \
              (np.max(lon, axis=1) > lonlim[0]) * (np.min(lon, axis=1) < lonlim[1])
valid_pixels = (np.max(lat, axis=0) > latlim[0]) * (np.min(lat, axis=0) < latlim[1]) * \
               (np.max(lon, axis=0) > lonlim[0]) * (np.min(lon, axis=0) < lonlim[1])
line_lim = [np.min(np.where(valid_lines == True)[0]), np.max(np.where(valid_lines == True)[0]) + 1]
pix_lim = [np.min(np.where(valid_pixels == True)[0]), np.max(np.where(valid_pixels == True)[0]) + 1]

# Save to disk
dat_size = (len(slc_dates), np.diff(line_lim)[0], np.diff(pix_lim)[0])
geo_size = (np.diff(line_lim)[0], np.diff(pix_lim)[0])

# Surfdrive folder to save data
folder = '/home/gert/Surfdrive/Florence_processing'

file_names = []
file_dtypes = []
file_sizes = []

# Create memmaps.
for name, dat in zip(['lat', 'lon', 'incidence_angle', 'dem'], [lat, lon, incidence_angle, dem]):
    filename = os.path.join(folder, name + '.raw')
    file_names.append('filename: ' + name + '.raw')
    data = np.memmap(filename=filename, dtype=np.float32, mode='w+', shape=geo_size)
    file_dtypes.append(' datatype: float32')
    file_sizes.append(' datasize: (' + str(geo_size[0]) + ', ' + str(geo_size[1]) + ')')
    data[:, :] = dat[line_lim[0]:line_lim[1], pix_lim[0]:pix_lim[1]]
    data.flush()

for name, dtype, data_type, conv_dtype, stack in zip(['slc', 'TOPS_phase_ramp'], [np.complex64, np.float32],
                                         ['complex64', 'float32'], ['complex_short', 'real4'], [slcs, ramps]):
    filename = os.path.join(folder, name + '.raw')
    file_names.append('filename: ' + name + '.raw')
    data = np.memmap(filename=filename, dtype=dtype, mode='w+', shape=dat_size)
    file_dtypes.append(' datatype: ' + data_type)
    file_sizes.append(' datasize: (' + str(dat_size[0]) + ', ' + str(geo_size[0]) + ', ' + str(geo_size[1]) + ')')
    for n, dat in enumerate(stack):
        data[n, :, :] = ImageData.disk2memory(dat[line_lim[0]:line_lim[1], pix_lim[0]:pix_lim[1]], conv_dtype)
    data.flush()

# Add text file
txt_file = os.path.join(folder, 'readme.txt')
f = open(txt_file, 'w+')

f.write('Data for ' + stack_name + '\n')
f.write('Dates in datastack' + '\n')
for date in slc_dates:
    f.write(date + '\n')

f.write('' + '\n')
f.write('Data files:' + '\n')
for file_name, file_dtype, file_size in zip(file_names, file_dtypes, file_sizes):
    f.write(file_name + file_dtype + file_size + '\n')

f.write('' + '\n')
f.write('Example code to read file' + '\n')
f.write('import numpy as np' + '\n')
f.write('import os' + '\n')
f.write('path_to_folder = "add your folder name here"' + '\n')
f.write('np.memmap(filename=os.path.join(path_to_folder, "slc.raw"), dtype=np.complex64, mode="r", shape=' +
        '(' + str(dat_size[0]) + ', ' + str(geo_size[0]) + ', ' + str(geo_size[1]) + ')' + '\n')
f.close()
