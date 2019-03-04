# This is a test_file to do the processing for one ECMWF date. This includes:
# - project_functions of ECMWF data
# - project_functions of SRTM data
# - creation of a radar DEM
# Because a DEM is generally already created, the script will ask for a username and password for the usgs website.
# You can create an account at .....

import matplotlib.pyplot as plt
from matplotlib import cm
import os
import numpy as np
import csv
import datetime

from rippl.NWP_functions.ECMWF.ecmwf_type import ECMWFType
from rippl.NWP_functions.ECMWF.ecmwf_download import ECMWFdownload
from rippl.NWP_functions.ECMWF.ecmwf_load_file import ECMWFData

from i import ImageData
from rippl.NWP_functions.model_ray_tracing import ModelRayTracing
from rippl.NWP_functions.radar_data import RadarData
from rippl.NWP_functions.model_to_delay import ModelToDelay
from rippl.NWP_functions.model_interpolate_delays import ModelInterpolateDelays

# path to your test data folder
track = '15'
asc_dsc = 'asc'

download_folder = '/media/gert/Data/weather_models/ecmwf_data'
radar_datastack = '/media/gert/Data/radar_results/netherlands/correction_ps_points/T' + track + '_network'
master_image = '/media/gert/Data/radar_results/netherlands/correction_ps_points/T' + track + '_network'
dem_folder = '/media/gert/Data/DEM/dem_processing'

# We project_functions data for western europe. (This can be extended or narrowed, but should at least include the satellite
# orbit and radar points on the ground.)
latlim = [45, 56]
lonlim = [-2, 12]

# You can choose three different ecmwf datasets:
# - era-interim > re-analysis from 1970 till now (delay few months) 0.75 degrees
# - era5 > re-analysis from 2010-2016, but will be extended for longer periods later 0.25 degrees
# - operational > archived opertational data from ecmwf data. (currently not possible to project_functions)
data_type = 'era5'

# Option between nearest (neighbour) and linear in time. Takes the closest or linear interpolated delay.
time_interp = 'nearest'

# Interval of radar pixels for delay calculations. Two points per model pixel should be fine.
# For example 0.25 degree > 20 km > 1000 radar pixels > less than 500 pixels is fine.
interval = [50, 200]
# Here you can specify whether you want the different parts of the delay calculated. If you are interested in the total
# delay only, you can set this to False. Otherwise also the hydrostatic, wet and liquid delay are calculated seperately.
split_signal = False

# Find and load all overpasses
overpasses_list = '/media/gert/Data/radar_results/netherlands/correction_ps_points/T' + track + '_network/nederland_s1_' + asc_dsc + '_t' + track + '_secondary_ps_atm_obs_time.txt'
f = open(overpasses_list, 'rb')
overpasses = [datetime.datetime.strptime(d[:-1], '%Y%m%dT%H%M') for d in f.readlines()]
all_overpasses = overpasses

first_overpass = datetime.datetime(year=2015, month=5, day=1)
last_overpass = datetime.datetime(year=2018, month=1, day=1)

# Take only part of dataset
overpasses = np.array([overpass for overpass in overpasses if (overpass < last_overpass and overpass > first_overpass)])

ecmwf_type = ECMWFType(data_type)
radar_data = RadarData(time_interp, ecmwf_type.t_step, interval)
for date in overpasses:
    radar_data.match_overpass_weather_model(date)

# Download the needed files
down = ECMWFdownload(latlim, lonlim, download_folder, data_type=data_type)
down.prepare_download(radar_data.date_times)
down.download()

# Create master dem grid.
meta = ImageData(os.path.join(master_image, 'master.res'), 'single')
radar_data.calc_geometry(dem_folder, meta=meta)

# Index the needed files
data = ECMWFData(ecmwf_type.levels)

# Initialize geometry for rac_tracing
ray_delays = ModelRayTracing(split_signal=False)
int_str = '_int_' + str(interval[0]) + '_' + str(interval[1]) + '_buf_' + str(interval[0]) + '_' + str(interval[1])
ray_delays.load_geometry(radar_data.lines, radar_data.pixels,
                         meta.data_memory['azimuth_elevation_angle']['Azimuth_angle' + int_str],
                         meta.data_memory['azimuth_elevation_angle']['Elevation_angle' + int_str],
                         meta.data_memory['geocode']['Lat' + int_str],
                         meta.data_memory['geocode']['Lon' + int_str],
                         meta.data_memory['radar_dem']['Data' + int_str])

# Import the ps points from csv file
csv_file = '/media/gert/Data/radar_results/netherlands/correction_ps_points/T' + track + '_network/nederland_s1_' + asc_dsc + '_t' + track + '_secondary_ps_atm.csv'
f = open(csv_file, 'rb')
csv_dat = csv.reader(f)

ps_line = []
ps_pixel = []
ps_height = []
i = 0
for ps in csv_dat:
    if i == 0:
        i = 1
        continue
    ps_line.append(int(ps[5]))
    ps_pixel.append(int(ps[6]))
    ps_height.append(float(ps[3]))

ps_pixel = np.array(ps_pixel)
ps_line = np.array(ps_line)

# Initialize the point delays
point_delays = ModelInterpolateDelays(radar_data.lines, radar_data.pixels, split=False)
point_delays.add_interp_points(ps_line, ps_pixel, np.array(ps_height) + 45.00)

for date, filename in zip(down.dates, down.filenames):

    time = date.strftime('%Y%m%dT%H%M')
    data.load_ecmwf(date, filename)

    # And convert the data to delays
    geoid_file = os.path.join(dem_folder, 'egm96.raw')

    model_delays = ModelToDelay(ecmwf_type.levels, geoid_file)
    model_delays.load_model_delay(data.model_data)
    model_delays.model_to_delay()

    # Convert model delays to delays over specific rays
    ray_delays.load_delay(model_delays.delay_data)
    ray_delays.calc_cross_sections()
    ray_delays.find_point_delays()

    # Finally, convert for specific points
    point_delays.add_delays(ray_delays.spline_delays)
    point_delays.interpolate_points()

    # Remove intermediate results
    data.remove_ecmwf(time)
    model_delays.remove_delay(time)
    ray_delays.remove_delay(time)

# Save as csv.
csv_file = os.path.join(radar_datastack, 'results', data_type + '_' + track + '.csv')

all_fields = [d.strftime('%Y%m%dT%H%M') for d in all_overpasses]
csv_file = open(csv_file, 'w')
writer = csv.DictWriter(csv_file, fieldnames=all_fields)

row_dict = dict()
for dat in all_fields:
    row_dict[dat] = 0

fields = [d.strftime('%Y%m%dT%H%M') for d, f in zip(all_overpasses, down.filenames) if f]
keys = sorted(point_delays.interp_delays['total'].keys())
for i in np.arange(len(ps_height)):

    print('Save row ' + str(i))

    for key, key_csv in zip(keys, fields):
        row_dict[key_csv] = point_delays.interp_delays['total'][key][i]

    writer.writerow(row_dict)

csv_file.close()

# Show images of ps delays
plt.ioff()

for i in np.arange(len(keys) - 1):
    dat_1 = keys[i]
    dat_2 = keys[i + 1]
    plt.figure()
    ifg = np.remainder(point_delays.interp_delays['total'][dat_2] - point_delays.interp_delays['total'][dat_1], 0.05546576 / 2) / 0.05546576 * np.pi * 4
    plt.scatter(-ps_pixel, -ps_line, c=ifg, cmap=cm.jet)
    plt.title('Difference in delay between ' + dat_1 + ' and ' + dat_2 + ' for ' + data_type + ' data.')
    plt.colorbar()
    im_file = os.path.join(radar_datastack, 'results', 'ifg_' + dat_1 + '_' + dat_2 + '_' + data_type + '.png')
    plt.savefig(im_file)
    plt.close()

    plt.scatter(-ps_pixel, -ps_line, c=point_delays.interp_delays['total'][dat_2] - point_delays.interp_delays['total'][dat_1], cmap=cm.jet)
    plt.title('Difference in delay between ' + dat_1 + ' and ' + dat_2 + ' for ' + data_type + ' data.')
    plt.colorbar()
    im_file = os.path.join(radar_datastack, 'results',
                           'ifg_' + dat_1 + '_' + dat_2 + '_' + data_type + '_unwrapped.png')
    plt.savefig(im_file)
    plt.close()
