script_path = '/usr/people/mulder/software/Gert'

import datetime
import getpass
import os
import re
import shutil
import sys
import time

import numpy as np

if '/usr/local/free/lib/python2.7/site-packages' in sys.path:
    sys.path.remove('/usr/local/free/lib/python2.7/site-packages')
if '/usr/local/free/lib/python2.7/site-packages/paramiko-1.15.0-py2.7.egg' in sys.path:
    sys.path.remove('/usr/local/free/lib/python2.7/site-packages/paramiko-1.15.0-py2.7.egg')
sys.path.extend([os.path.join(script_path, 'doris_processing/sentinel')])
from read_weather_stations import ifgs_delay_coor
from create_harmonie_grid import read_grib_coor
from multilook import multilook
from image_metadata import ResData
from calculate_slant import prepare_slant_input
from additional_data import save_metadata

server = True
host = 'hpc03.tudelft.net'
username = 'gertmulder'
type = 'image_folder'
network = True
convert = True

filenames = ['coherence.ras', 'cint.raw_pha.ras', 'unwrapped.ras', 'unwrapped.raw', 'coherence_ml.raw', 'cint_filt_ml.raw', 'ifgs.res', 'master.res', 'slave.res']
multilook_files = ['coherence_ml.raw', 'unwrapped.raw', 'cint_filt_ml.raw']
base_cluster = '/home/gertmulder/datastacks'
#base_folder = '/usr/people/mulder/data/sar_results'
base_folder = '/media/gert/Data/radar_data/'
#base_folder = '/Users/gertmulder/surfdrive/TU_Delft/radar_data'
tracks = ['netherlands/nl_full_t037'] # , 'netherlands/nl_full_t110', 'netherlands/nl_full_t088_vv',
          # 'netherlands/nl_full_t088_vh', 'europe/nl_france_t037']
overpass_time = ['05:50']  #, '05:58', '17:24', '17:24', '5:50']
orbit_folder = '/media/gert/Data/orbits/sentinel-1'

password = getpass.getpass("Enter your password")

# resample the dem / phi / lam  if needed.
range_orig = 40
az_orig = 10

# comparable to 1km, 2km, 5km and 10km pixels.
ranges_s = [40, 200, 400, 1000]
azs_s = [10, 50, 100, 250]
gridsizes_s = ['200m', '1km', '2km', '5km']

ranges = [200, 400, 1000]
azs = [50, 100, 250]
gridsizes = ['1km', '2km', '5km']

ranges_l = [400, 1000]
azs_l = [100, 250]
gridsizes_l = ['2km', '5km']


for overpass, track in zip(overpass_time, tracks):
    stack_folder = os.path.join(base_cluster, track)
    output_folder = os.path.join(base_folder, track)

    #for filename in filenames:
        #find_paths(stack_folder, output_folder, filename, server, host, username, password, type, network, convert)

    # Now find the different dates and project_functions some additional files.
    folders = next(os.walk(output_folder))[1]

    # First multilook the dem / phi / lam if needed
    files = ['dem_radar', 'lam', 'phi']

    for ml_range, ml_az, gridsize in zip(ranges_s, azs_s, gridsizes_s):
        # Read size from master.res
        res_name = os.path.join(output_folder, 'dem_lat_lon_orbit', 'master.res')
        res_dat = ResData(res_name)
        pixels = int(res_dat.processes['readfiles']['Number_of_pixels_original'])

        for filename in files:
            in_file = os.path.join(output_folder, 'dem_lat_lon_orbit', filename + '.raw')
            out_file = os.path.join(output_folder, 'dem_lat_lon_orbit', filename + '_' + gridsize + '.raw')
            if not os.path.exists(out_file):
                multilook(in_file, out_file, az=ml_az, ra=ml_range, step='dem')

    for ml_range, ml_az, gridsize in zip(ranges_s, azs_s, gridsizes_s):
        # Then calculate the zenith angles based on orbit and lat / lon values.
        phi_in = 'phi_' + gridsize + '.raw'
        lam_in = 'lam_' + gridsize + '.raw'
        dem_in = 'dem_radar_' + gridsize + '.raw'
        harmonie_txt_file = 'coor_' + gridsize + '.txt'
        heading_out = 'heading_' + gridsize + '.raw'
        inclination_out = 'inclination_' + gridsize + '.raw'

        os.chdir(os.path.join(output_folder, 'dem_lat_lon_orbit'))

        if not os.path.exists(harmonie_txt_file) or not os.path.exists(heading_out) or not os.path.exists(inclination_out):
            heading, inclination = prepare_slant_input(orbit_dir=orbit_folder, phi_dat=phi_in, lam_dat=lam_in, dem_dat=dem_in,
                                ifg_folder=os.path.join(output_folder, 'dem_lat_lon_orbit'), ml_az=ml_az, ml_ra=ml_range,
                                outfile=harmonie_txt_file, head_dat=heading_out, incl_dat=inclination_out)

    dates = []
    ifgs = []
    for folder in folders:
        match = re.search(r'[\d\d\d\d\d\d\d\d]+_[\d\d\d\d\d\d\d\d]+', folder)
        if match:
            date_1 = datetime.datetime.strptime(folder[:8] + overpass, "%Y%m%d%H:%M")
            date_2 = datetime.datetime.strptime(folder[9:17] + overpass, "%Y%m%d%H:%M")

            if not date_1 in dates:
                dates.append(date_1)
            if not date_2 in dates:
                dates.append(date_2)
            if not [date_1, date_2] in ifgs:
                ifgs.append([date_1, date_2])

    for date in dates:
        folder = os.path.join(output_folder, date.strftime('%Y%m%d'))
        if not os.path.exists(folder):
            os.mkdir(folder)

        save_metadata(folder, date)

        # Create delay map for Harmonie



    for ifg in ifgs:
        folder = os.path.join(output_folder, ifg[0].strftime('%Y%m%d') + '_' + ifg[1].strftime('%Y%m%d') + '_ifg')
        if not os.path.exists(folder):
            os.mkdir(folder)

        for date in ifg:
            save_metadata(folder, date)

        # First define size of grid based on harmonie grid
        latlim, lonlim, dlat, dlon = read_grib_coor(
            filename='/home/gert/surfdrive/TU_Delft/Data KNMI/data_voor_gert.grib', border=50, n=10)

        gridsize = '200m'
        lat = os.path.join(output_folder, 'dem_lat_lon_orbit', 'phi_' + gridsize + '.raw')
        lon = os.path.join(output_folder, 'dem_lat_lon_orbit', 'lam_' + gridsize + '.raw')
        dem = os.path.join(output_folder, 'dem_lat_lon_orbit', 'dem_radar_' + gridsize + '.raw')
        incl = os.path.join(output_folder, 'dem_lat_lon_orbit', 'inclination_' + gridsize + '.raw')
        res_m = ResData(filename=os.path.join(output_folder, folder, 'master.res'), type='single')

        lines = int(res_m.processes['readfiles']['Number_of_lines_original']) / 10
        pixels = int(res_m.processes['readfiles']['Number_of_pixels_original']) / 40

        lat_dat = np.memmap(lat, dtype='float32', mode='r', shape=(lines, pixels)).astype(dtype='float32',subok=False)
        lon_dat = np.memmap(lon, dtype='float32', mode='r', shape=(lines, pixels)).astype(dtype='float32',subok=False)
        dem_dat = np.memmap(dem, dtype='float32', mode='r', shape=(lines, pixels)).astype(dtype='float32',subok=False)
        incl_dat = np.memmap(incl, dtype='float32', mode='r', shape=(lines, pixels)).astype(dtype='float32',subok=False)

        output_file = os.path.join(output_folder, folder)

        time_1 = ifg[0].strftime('%Y%m%d%H%M')
        time_2 = ifg[1].strftime('%Y%m%d%H%M')
        ifgs_delay_coor(time_1, time_2, lat=lat_dat, lon=lon_dat, elevation=dem_dat, incl=incl_dat, plot=True, file_dir=output_file)


    for ifg in ifgs:
        folder = os.path.join(output_folder, ifg[0].strftime('%Y%m%d') + '_' + ifg[1].strftime('%Y%m%d') + '_ifg')
        if not os.path.exists(folder):
            os.mkdir(folder)
        os.chdir(folder)

        if os.path.exists(os.path.join(folder, 'cint_filt_ml.raw')) and os.path.exists(os.path.join(folder, 'coherence_ml.raw')):
            for ml_range, ml_az, gridsize in zip(ranges, azs, gridsizes):

                shutil.copy('ifgs.res', 'ifgs_old.res')

                # apply multilook for filtphase and coherence
                new_dat = 'cint_filt_' + gridsize + '.raw'
                multilook('cint_filt_ml.raw', new_dat, az=ml_az, ra=ml_range, step='filtphase')

                new_coh = 'coherence_' + gridsize + '.raw'
                multilook('coherence_ml.raw', new_coh, az=ml_az, ra=ml_range, step='coherence')

                inputfile_loc = os.path.join(os.path.dirname(os.path.dirname(output_folder)), 'input_files', 'input.unwrap')

                ifgs_dat = ResData('ifgs.res', 'ifgs')
                if ifgs_dat.process_control['unwrap'] == '1':
                    ifgs_dat.del_process('unwrap')
                    ifgs_dat.write()

                shutil.copy(new_coh, 'unwrap_input.raw')

                # unwrap again for multilooked images.
                command = 'doris ' + inputfile_loc
                os.system(command)
                time.sleep(2)

                # copy old ifgs file back.
                if os.path.exists('unwrapped.raw'):
                    shutil.move('unwrapped.raw', 'unwrapped_' + gridsize + '.raw')
                if os.path.exists('unwrapped_interferogram.ras'):
                    shutil.move('unwrapped_interferogram.ras', 'unwrapped_' + gridsize + '.ras')
                shutil.copy('ifgs_old.res', 'ifgs.res')

    for ifg in ifgs:
        for ml_range, ml_az, gridsize in zip(ranges_l, azs_l, gridsizes_l):
            # create network for low resolution
            # comparable to 1km, 2km, 5km and 10km pixels.

            print('First finish the other parts!')



