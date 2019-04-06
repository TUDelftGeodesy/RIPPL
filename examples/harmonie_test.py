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

from i import ImageData
from rippl.NWP_simulations.model_ray_tracing import ModelRayTracing
from rippl.NWP_simulations.radar_data import RadarData
from rippl.NWP_simulations.model_to_delay import ModelToDelay
from rippl.NWP_simulations.model_interpolate_delays import ModelInterpolateDelays
from rippl.NWP_simulations.harmonie.harmonie_database import HarmonieDatabase
from rippl.NWP_simulations.harmonie.harmonie_load_file import HarmonieData

# path to your test data folder
track = '110'
asc_dsc = 'dsc'

# path to your test data folder
archive_folder = '/media/gert/Data/weather_models/harmonie_data'
download_folder = '/media/gert/Data/weather_models/ecmwf_data'
radar_datastack = '/media/gert/Data/radar_results/netherlands/correction_ps_points/T' + track + '_network'
master_image = '/media/gert/Data/radar_results/netherlands/correction_ps_points/T' + track + '_network'
dem_folder = '/media/gert/Data/DEM/dem_processing'

# Option between nearest (neighbour) and linear in time. Takes the closest or linear interpolated delay.
time_interp = 'nearest'

# Interval of radar pixels for delay calculations. Two points per model pixel should be fine.
# For example 0.25 degree > 20 km > 1000 radar pixels > less than 500 pixels is fine.
interval = [50, 200]
# Here you can specify whether you want the different parts of the delay calculated. If you are interested in the total
# delay only, you can set this to False. Otherwise also the hydrostatic, wet and liquid delay are calculated seperately.
split_signal = True

# Find and load all overpasses
overpasses_list = '/media/gert/Data/radar_results/netherlands/correction_ps_points/T' + track + '_network/nederland_s1_' + asc_dsc + '_t' + track + '_secondary_ps_atm_obs_time.txt'
f = open(overpasses_list, 'rb')
overpasses = [datetime.datetime.strptime(d[:-1], '%Y%m%dT%H%M') for d in f.readlines()]
all_overpasses = overpasses

# Take only part of dataset
overpasses = np.array(overpasses)

harmonie_archive = HarmonieDatabase(database_folder=archive_folder)
dates = []
filenames = []

for overpass in overpasses:
    filename, date = harmonie_archive(overpass)
    dates.append(date[0])
    filenames.append(filename[0])

# Create master dem grid.
meta = ImageData(os.path.join(master_image, 'master.res'), 'single')
radar_data = RadarData(time_interp, datetime.timedelta(minutes=15), interval)
radar_data.calc_geometry(dem_folder, meta=meta)

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

# Adjust the range and azimuth times.
ps_pixel = np.array(ps_pixel)
ps_line = np.array(ps_line)

# Initialize the point delays
point_delays = ModelInterpolateDelays(radar_data.lines, radar_data.pixels, split=False)
point_delays.add_interp_points(ps_line, ps_pixel, np.array(ps_height) + 45.00)

# Initialize load harmonie data
data = HarmonieDatabase()

for date, filename in zip(dates, filenames):

    if filename:
        time = date.strftime('%Y%m%dT%H%M')
        data.load_harmonie(date, filename)

        # And convert the data to delays
        geoid_file = os.path.join(dem_folder, 'egm96.raw')

        model_delays = ModelToDelay(65, geoid_file)
        model_delays.load_model_delay(data.model_data)
        model_delays.model_to_delay()
        model_delays.delay_data[time]['total_delay'] = np.flip(model_delays.delay_data[time]['total_delay'], axis=1)

        # Convert model delays to delays over specific rays
        ray_delays.load_delay(model_delays.delay_data)
        ray_delays.calc_cross_sections()
        ray_delays.find_point_delays()

        # Finally, convert for specific points
        point_delays.add_delays(ray_delays.spline_delays)
        point_delays.interpolate_points()

        # Remove intermediate results
        data.remove_harmonie(time)
        model_delays.remove_delay(time)
        ray_delays.remove_delay(time)

# Save as csv.
csv_file = os.path.join(radar_datastack, 'results', 'harmonie_' + track + '.csv')

all_fields = [d.strftime('%Y%m%dT%H%M') for d in all_overpasses]
csv_file = open(csv_file, 'w')
writer = csv.DictWriter(csv_file, fieldnames=all_fields)

row_dict = dict()
for dat in all_fields:
    row_dict[dat] = 0

fields = [d.strftime('%Y%m%dT%H%M') for d, f in zip(all_overpasses, filenames) if f]
keys = sorted(point_delays.interp_delays['total'].keys())
for i in range(len(ps_height)):

    print('Save row ' + str(i))

    for key, key_csv in zip(keys, fields):
        row_dict[key_csv] = point_delays.interp_delays['total'][key][i]

    writer.writerow(row_dict)

csv_file.close()

# Show images of ps delays
plt.ioff()

for i in range(len(keys) - 1):
    dat_1 = keys[i]
    dat_2 = keys[i + 1]
    plt.figure()
    ifg = np.remainder(point_delays.interp_delays['total'][dat_2] - point_delays.interp_delays['total'][dat_1], 0.05546576 / 2) / 0.05546576 * np.pi * 4
    plt.scatter(-ps_pixel, -ps_line, c=ifg, cmap=cm.jet)
    plt.title('Difference in delay between ' + dat_1 + ' and ' + dat_2 + ' for harmonie data.')
    plt.colorbar()
    im_file = os.path.join(radar_datastack, 'results', 'ifg_' + dat_1 + '_' + dat_2 + '_' + 'harmonie.png')
    plt.savefig(im_file)
    plt.close()

    plt.scatter(-ps_pixel, -ps_line, c=point_delays.interp_delays['total'][dat_2] - point_delays.interp_delays['total'][dat_1], cmap=cm.jet)
    plt.title('Difference in delay between ' + dat_1 + ' and ' + dat_2 + ' for harmonie data.')
    plt.colorbar()
    im_file = os.path.join(radar_datastack, 'results', 'ifg_' + dat_1 + '_' + dat_2 + '_' + 'harmonie_unwrapped.png')
    plt.savefig(im_file)
    plt.close()
