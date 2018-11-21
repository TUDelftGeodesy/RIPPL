# This is a test_file to do the processing for one ECMWF date. This includes:
# - project_functions of ECMWF data
# - project_functions of SRTM data
# - creation of a radar DEM
# Because a DEM is generally already created, the script will ask for a username and password for the usgs website.
# You can create an account at .....

import os
import numpy as np
import csv
import datetime

from NWP_functions.ECMWF.ecmwf_type import ECMWFType
from NWP_functions.ECMWF.ecmwf_download import ECMWFdownload
from NWP_functions.ECMWF.ecmwf_load_file import ECMWFData

from image_data import ImageData
from NWP_functions.model_ray_tracing import ModelRayTracing
from NWP_functions.radar_data import RadarData
from NWP_functions.model_to_delay import ModelToDelay
from NWP_functions.model_interpolate_delays import ModelInterpolateDelays

def ecmwf_run_ps(download_folder, dem_folder, reference_orbit, ps_points, dates, interval='', data_type='era5',
                 time_interp='nearest', latlim='', lonlim='', split_signal=False):

    if len(interval) != 2:
        interval = [50, 200]
    if len(latlim) != 2:
        latlim = [45, 56]
    if len(lonlim) != 2:
        lonlim = [-2, 12]

    ecmwf_type = ECMWFType(data_type)
    radar_data = RadarData(time_interp, ecmwf_type.t_step, interval)
    for date in dates:
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

    out_dat = dict()
    keys = sorted(point_delays.interp_delays['total'].keys())

    for key in keys:
        out_dat[key] = point_delays.interp_delays['total'][key][i]

    return out_dat

