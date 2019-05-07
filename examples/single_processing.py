#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# This is an low level RIPPL example code, each processing step is carried out for one burst (master and slave) separately.
#
# Gtiff output NOT implemented
#
# The script is structured as followed:
########################################################################################################################
# Prepare Data:
#     1. download precise orbit files
#     2. Read in / Create Sentinel 1 stack
#     3. download SRTM data for S1 stack
#
#     4. Assign master and slave (method 1: read from .res file, method 2: load ImageData Object)
#     5. Create Coordinate systems (Radar & DEM coordinate system and output CRS (includes multilooking))
#
# Processing Flow
#     6. create SRTM DEM
#     7. InverseGeocode                (master)
#     8. RadarDEM                      (master)
#     9. Geocode                       (master)
#     10. AzimuthElevationAngle        (master)
#     11. GeometricalCoreg             (master and slave)
#     12. Deramp                       (slave)
#     13. Resample                     (slave)
#     14. Reramp                       (slave)
#     15. EarthTopoPhase               (slave)
#     16. SquareAmplitude              (for master and slave)
#
#
#
# Create final products
#     17. Interferogram
#     18. Multilooking                 (master and slave)
#     19. Coherence
#     20. Unwrap
########################################################################################################################
# Important Parameters to run functions:
#
# meta		    = ImageData object
# cmaster	    = coreg master
# coordinates	= CoordinateSystem object, output coordinate system
#
# file_type	    = processing step (e.g. ‘deramp’) or output product (e.g. '['lat', 'lon', 'X', 'Y', 'Z']')
# 		          possible flags can be found in function (add_meta_data)
#
# step		    = processing step the function is using (e.g. use 'earth_topo_phase' for creating interferogram)
########################################################################################################################

import sys
#sys.path.append('/home/mmanne/Documents/SCRIPTING/Python')
#sys.path.remove('/home/mmanne/Documents/SCRIPTING/Python/rippl')

# load RIPPL
from rippl.stack import Stack
from rippl.SAR_sensors.sentinel.sentinel_stack import SentinelStack
from rippl.coordinate_system import CoordinateSystem
import os

parallel = True
n_jobs = 7
# Number of cores
cores = 7

track_no = 8

# define required input parameters
# AIO Spain
start_date = '2018-09-03'
end_date = '2018-10-29'
master_date = '2018-09-21'
database_folder = '/mnt/DATA_LOCAL/Test_Data/Data/Spain/Zaragoza_SLC/new/s1_dec_t8'
stack_folder = '/mnt/DATA_LOCAL/Test_Data/Data/Spain/t8_burst/processed'
polarisation = 'VV'
mode = 'IW'
product_type = 'SLC'
track_no = 8

srtm_folder = '/mnt/DATA_LOCAL/Test_Data/Data/Spain/t8_burst/SRTM'
orbit_folder = '/mnt/DATA_LOCAL/Test_Data/Data/Spain/Zaragoza_Orbits/t8'
shapefile = '/mnt/DATA_LOCAL/Test_Data/Data/Spain/Zaragoza_AOI/AB_ROI_copy.shp'

# 1. Download precise orbit files
#precise_folder = os.path.join(orbit_folder, 'precise')
#download_orbit = DownloadSentinelOrbit(start_date, end_date, precise_folder)
#download_orbit.download_orbits()

# 2.  Creates a Sentinel 1 stack, if already existing read in existing stack
if not os.listdir(stack_folder):
    s1_stack = SentinelStack(stack_folder)
    s1_stack.read_from_database(database_folder, shapefile, track_no, orbit_folder, start_date, end_date, master_date,
                             mode, product_type, polarisation, cores=6)
    s1_stack = Stack(stack_folder)
    s1_stack.read_master_slice_list()
    s1_stack.read_stack(start_date, end_date)
    s1_stack.add_master_res_info()
    # define interferogram(network_type: temp_baseline, daisy_chain, single_master)
    s1_stack.create_network_ifgs(network_type='temp_baseline', temp_baseline=6)
else:
    s1_stack = Stack(stack_folder)
    s1_stack.read_master_slice_list()
    s1_stack.read_stack(start_date, end_date)

# 3. download SRTM data
password = 'Radar2016'
username = 'gertmulder'
s1_stack.create_SRTM_input_data(srtm_folder, username, password, srtm_type='SRTM3')


# 4. assign / load data (two methods available)
from rippl.image_data import ImageData
load_data_method = 2

if load_data_method == 1:
    # method one: read date from res file
    master_burst = ImageData('/mnt/DATA_LOCAL/Test_Data/Data/Namib_Desert/Namib_processed/burst_level/20180922/slice_501_swath_1_VV/info.res', 'single')
    slave_burst = ImageData('/mnt/DATA_LOCAL/Test_Data/Data/Namib_Desert/Namib_processed/burst_level/20180910/slice_501_swath_1_VV/info.res', 'single')
elif load_data_method == 2:
    # method two: get data from a ImageDate object (saved on disk)
    s1_stack.images['20180921'].load_slice_info()
    s1_stack.images['20180927'].load_slice_info()
    # assign slice
    master_burst = s1_stack.images['20180921'].slices['slice_501_swath_1_VV']
    slave_burst = s1_stack.images['20180927'].slices['slice_501_swath_1_VV']

# load the memmap files from disk
master_burst.read_data_memmap()
slave_burst.read_data_memmap()

# assign ifg_meta data to variable
ifg_meta = s1_stack.interferograms['20180921_20180927'].res_data

# assign cmaster
cmaster = s1_stack.interferograms['20180921_20180927'].cmaster

# 5. Create / Define Coordinate Systems for data, CRS are needed for projection conversion between
# radar, geographic and projected coordinates systems. A new coordinate system is also needed for multilooking

# assign a output radar coordinate system
coordinates = CoordinateSystem()
coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
# on slice level
coordinates.slice = True

# assign a geographic coordinate system for DEM that is used for the InverseGeocoding step
# here read coordinate system from res_file (processingstep 'import_DEM')
coordinates_DEM = master_burst.read_res_coordinates('import_DEM')[0]

# assign a new coordinate system as output CRS
coordinates_multi = CoordinateSystem()
coordinates_multi.create_radar_coordinates(master_burst, multilook=[10, 10], offset=[0, 0], oversample=[1, 1])
# on slice level
coordinates_multi.slice = True

# 6. create SRTM DEM for processing
# load function
from rippl.processing_steps.import_dem import CreateSrtmDem
# assign function with input parameters
Srtm_step = CreateSrtmDem(master_burst, srtm_folder=srtm_folder)
# add meta data
Srtm_step.add_meta_data(master_burst, coordinates_DEM)
# create empty output
Srtm_step.create_output_files(master_burst, 'DEM', coordinates_DEM)
# execute function
Srtm_step()
# save result to disk
Srtm_step.save_to_disk(master_burst, 'DEM', coordinates_DEM)
# clear result from memory
# Srtm_step.clear_memory(master_burst)

# 7. InverseGeocode
from rippl.processing_steps.inverse_geocode import InverseGeocode
invers_step = InverseGeocode(master_burst, coordinates_DEM)
invers_step.add_meta_data(master_burst, coordinates_DEM)
# here creates output DEM_line.raw and DEM_pixel.raw
invers_step.create_output_files(master_burst, ['DEM_line', 'DEM_pixel'], coordinates_DEM)
invers_step()
invers_step.save_to_disk(master_burst, ['DEM_line', 'DEM_pixel'], coordinates_DEM)

# 8. Radar DEM
from rippl.processing_steps.radar_dem import RadarDem
radar_dem_step = RadarDem(master_burst, coordinates, coor_in=coordinates_DEM)
radar_dem_step.add_meta_data(master_burst, coordinates)
radar_dem_step.create_output_files(master_burst, 'DEM', coordinates)
radar_dem_step()
radar_dem_step.save_to_disk(master_burst, 'DEM', coordinates)

# 9. Geocode
from rippl.processing_steps.geocode import Geocode
geocode_step = Geocode(master_burst, coordinates)
geocode_step.add_meta_data(master_burst, coordinates)
geocode_step.create_output_files(master_burst, ['lat', 'lon', 'X', 'Y', 'Z'], coordinates)
geocode_step()
geocode_step.save_to_disk(master_burst, ['lat', 'lon', 'X', 'Y', 'Z'], coordinates)


# 10. AzimuthElevationAngle (optional)
from rippl.processing_steps.azimuth_elevation_angle import AzimuthElevationAngle
azi_ele_step = AzimuthElevationAngle(master_burst, coordinates)
azi_ele_step.add_meta_data(master_burst, coordinates)
azi_ele_step.create_output_files(master_burst, ['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], coordinates)
azi_ele_step()
azi_ele_step.save_to_disk(master_burst, ['elevation_angle', 'off_nadir_angle', 'heading', 'azimuth_angle'], coordinates)

# 11. GeometricalCoreg
from rippl.processing_steps.geometrical_coreg import GeometricalCoreg
geo_coreg_slave_step = GeometricalCoreg(master_burst, slave_burst, coordinates)
geo_coreg_slave_step.add_meta_data(slave_burst, master_burst, coordinates)
geo_coreg_slave_step.create_output_files(slave_burst, ['new_line', 'new_pixel'], coordinates)
geo_coreg_slave_step()
geo_coreg_slave_step.save_to_disk(slave_burst, ['new_line', 'new_pixel'], coordinates)

#12. Deramp
from rippl.processing_steps.deramping_reramping import Deramp
deramp_step = Deramp(slave_burst, coordinates)
deramp_step.add_meta_data(slave_burst)
deramp_step.create_output_files(slave_burst, 'deramp', coordinates)
deramp_step()
deramp_step.save_to_disk(slave_burst, 'deramp', coordinates)

# 13. Resample
from rippl.processing_steps.resample import Resample
resample_step = Resample(master_burst, slave_burst, coordinates)
resample_step.add_meta_data(slave_burst, coordinates)
resample_step.create_output_files(slave_burst, 'resample', coordinates)
resample_step()
resample_step.save_to_disk(slave_burst, 'resample', coordinates)

# 14. Reramp
from rippl.processing_steps.deramping_reramping import Reramp
reramp_step = Reramp(slave_burst, coordinates)
reramp_step.add_meta_data(slave_burst, coordinates)
reramp_step.create_output_files(slave_burst, 'reramp', coordinates)
reramp_step()
reramp_step.save_to_disk(slave_burst, 'reramp', coordinates)

# 15. EarthTopoPhase
from processing_steps.earth_topo_phase import EarthTopoPhase
earth_topo_step = EarthTopoPhase(slave_burst, coordinates, input_step='reramp')
earth_topo_step.add_meta_data(slave_burst, coordinates)
earth_topo_step.create_output_files(slave_burst, 'earth_topo_phase', coordinates)
earth_topo_step()
earth_topo_step.save_to_disk(slave_burst, 'earth_topo_phase', coordinates)

# 16. SquareAmplitude for master and slve
from processing_steps.square_amplitude import SquareAmplitude
# master
sqrt_ampt_master_step = SquareAmplitude(master_burst, coordinates, step='earth_topo_phase', file_type='earth_topo_phase')
sqrt_ampt_master_step.add_meta_data(master_burst, coordinates, step='earth_topo_phase', file_type='earth_topo_phase')
sqrt_ampt_master_step.create_output_files(master_burst, 'square_amplitude', coordinates)
sqrt_ampt_master_step()
sqrt_ampt_master_step.save_to_disk(master_burst, 'square_amplitude', coordinates)
# slave
sqrt_ampt_slave_step = SquareAmplitude(slave_burst, coordinates, step='earth_topo_phase', file_type='earth_topo_phase')
sqrt_ampt_slave_step.add_meta_data(slave_burst, coordinates, step='earth_topo_phase', file_type='earth_topo_phase')
sqrt_ampt_slave_step.create_output_files(slave_burst, 'square_amplitude', coordinates)
sqrt_ampt_slave_step()
sqrt_ampt_slave_step.save_to_disk(slave_burst, 'square_amplitude', coordinates)

# create output data
# 17. Interferogram
from processing_steps.interfero import Interfero
interfero_step = Interfero(slave_burst, master_burst, coordinates_multi, coor_in=coordinates, cmaster_meta=cmaster, ifg_meta=ifg_meta, step='earth_topo_phase', file_type='earth_topo_phase')
interfero_step.add_meta_data(ifg_meta, coordinates_multi, step='earth_topo_phase')
interfero_step.create_output_files(ifg_meta, 'interferogram', coordinates_multi)
interfero_step()
interfero_step.save_to_disk(ifg_meta, 'interferogram', coordinates_multi)

# 18. Multilook master and slave
from processing_steps.multilook import Multilook
# master
multi_master_step = Multilook(master_burst, master_burst, 'square_amplitude', 'square_amplitude', coordinates_multi, coor_in=coordinates)
multi_master_step.add_meta_data(master_burst, coordinates_multi, coordinates, 'square_amplitude', 'square_amplitude')
multi_master_step.create_output_files(master_burst, 'square_amplitude', 'square_amplitude', coordinates_multi)
multi_master_step()
multi_master_step.save_to_disk(master_burst, 'square_amplitude', 'square_amplitude', coordinates_multi)
# slave
multi_slave_step = Multilook(slave_burst, master_burst, 'square_amplitude', 'square_amplitude', coordinates_multi, coor_in=coordinates)
multi_slave_step.add_meta_data(slave_burst, coordinates_multi, coordinates, 'square_amplitude', 'square_amplitude')
multi_slave_step.create_output_files(slave_burst, 'square_amplitude', 'square_amplitude', coordinates_multi)
multi_slave_step()
multi_slave_step.save_to_disk(slave_burst, 'square_amplitude', 'square_amplitude', coordinates_multi)

# 19. Coherence
from processing_steps.coherence import Coherence
coherence_step = Coherence(slave_burst, master_burst, ifg_meta, coordinates_multi)
coherence_step.add_meta_data(ifg_meta, coordinates_multi)
coherence_step.create_output_files(ifg_meta, 'coherence', coordinates_multi)
coherence_step()
coherence_step.save_to_disk(ifg_meta, 'coherence', coordinates_multi)

# 20. Unwrap
from processing_steps.unwrap import Unwrap
unwrap_step = Unwrap(ifg_meta, coordinates_multi, step='interferogram', file_type='interferogram')
unwrap_step.add_meta_data(ifg_meta, coordinates_multi)
unwrap_step.create_output_files(ifg_meta, 'unwrap', coordinates_multi)
unwrap_step()
unwrap_step.save_to_disk(ifg_meta, 'unwrap', coordinates_multi)

# plot burst
import numpy as np
import matplotlib.pyplot as plt

ifg_data = interfero_step.ifg.data_disk['interferogram']['interferogram_ml_10_10']
plt.figure(1)
plt.imshow(np.angle(ifg_data)), plt.colorbar()
plt.show()

coh_data = coherence_step.coherence
plt.figure(2)
plt.imshow(coh_data), plt.colorbar()
plt.show()

amp_master = master_burst.data_disk['square_amplitude']['square_amplitude_ml_10_10']
plt.figure(3)
plt.imshow(np.log10(amp_master)), plt.colorbar()
plt.show()

amp_slave = slave_burst.data_disk['square_amplitude']['square_amplitude_ml_10_10']
plt.figure(4)
plt.imshow(np.log10(amp_master)), plt.colorbar()
plt.show()