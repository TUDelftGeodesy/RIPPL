"""
This class contains a full doris datastack. This links also to several functions which can be solved based on a
full stack approach. This is also the basis from which we create network of interferograms, which can be used for
different purposes.

Examples of functions are:
- network solution for ESD
- network solution for Ionosphere (or combined with ESD)
- sbas or squisar solutions for the whole network
- network solution by including harmonie data or ECMWF data with the interferograms

"""

import os
from image import Image
from interferogram import Interferogram
from orbit_dem_functions.srtm_download import SrtmDownload
from coordinate_system import CoordinateSystem
from processing_steps.import_dem import CreateSrtmDem
import datetime
import numpy as np


class Stack(object):

    def __init__(self, datastack_folder):
        self.datastack_folder = datastack_folder

        # List of images and interferograms
        self.images = dict()
        self.image_dates = []
        self.interferograms = dict()
        self.ifg_dates = []

        self.dates = []

        # Oversight of the different interferograms and images
        self.ifg_matrix = []
        self.baseline_matrix = []

        # master
        self.master_date = ''
        self.slice_names = []

        # Specific information master slices
        self.master_slice_swath_no = []
        self.master_slice_pol = []
        self.master_slice_lat = []
        self.master_slice_lon = []
        self.master_slice_date = []
        self.master_slice_az_time = []
        self.master_slice_x = []
        self.master_slice_y = []
        self.master_slice_z = []
        self.master_slice_time = []
        self.master_slice_seconds = []

        # Finally also give the slice numbers (we will start with 500 so we can count down if needed.)
        self.master_slice_number = []
        self.master_slice_names = []
        self.slice_number = []

    def read_master_slice_list(self):
        # az_time, yyyy-mm-ddThh:mm:ss.ssssss, swath x, slice i, x xxxx, y yyyy, z zzzz, lat ll.ll, lon ll.ll, pol pp

        list_file = os.path.join(self.datastack_folder, 'master_slice_list')
        if not os.path.exists(list_file):
            print('No existing master slices list found')
            return
        if len(self.master_slice_az_time) != 0:
            print('First read the list with master slices before reading in new data!')
            return

        l = open(list_file, 'r+')
        for line in l:
            sl = line.split(',')
            time = sl[0].split(' ')[1]
            t = datetime.datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f')
            self.master_slice_az_time.append(time)
            self.master_slice_time.append(t)
            self.master_slice_date.append(sl[0].split(' ')[1][:10])
            self.master_slice_swath_no.append(int(sl[1].split(' ')[1]))
            self.master_slice_number.append(int(sl[2].split(' ')[1]))
            self.master_slice_x.append(int(sl[3].split(' ')[1]))
            self.master_slice_y.append(int(sl[4].split(' ')[1]))
            self.master_slice_z.append(int(sl[5].split(' ')[1]))
            self.master_slice_lat.append(float(sl[6].split(' ')[1]))
            self.master_slice_lon.append(float(sl[7].split(' ')[1]))
            self.master_slice_pol.append(sl[8].split(' ')[1])
            self.master_slice_seconds.append(float(t.hour * 3600 + t.minute * 60 + t.second) + float(t.microsecond) / 1000000)

            self.master_slice_names.append('slice_' + sl[2].split(' ')[1] +
                                    '_swath_' + sl[1].split(' ')[1] + '_' + sl[8].split(' ')[1])
            self.master_date = t.strftime('%Y%m%d')

        l.close()

    def read_stack(self, first_date='1900-01-01', last_date='2100-01-01'):
        # This function reads the whole stack in memory. A stack consists of:
        # - images > with individual slices (yyyymmdd)
        # - interferograms > with individual slices if needed. (yyyymmdd_yyyymmdd)
        # First date and last give the maximum and minimum date to load (in case we want to load only a part of the stack.
        # Note: The master date is always loaded!

        start = int(first_date[:4] + first_date[5:7] + first_date[8:10])
        end = int(last_date[:4] + last_date[5:7] + last_date[8:10])

        dirs = next(os.walk(self.datastack_folder))[1]
        images = sorted([os.path.join(self.datastack_folder, x) for x in dirs if (len(x) == 8 and
                         start <= int(x) <= end) or x == self.master_date])
        ifgs = sorted([os.path.join(self.datastack_folder, x) for x in dirs if len(x) == 17 and
                       start <= int(x[:8]) <= end and start <= int(x[9:]) <= end])

        for im_dir in images:
            image_dir = os.path.join(self.datastack_folder, im_dir)

            if im_dir not in self.image_dates:
                self.images[im_dir[-8:]] = Image(image_dir, self.master_slice_names)
                self.image_dates.append(im_dir[-8:])

        cmaster_image = self.images[self.master_date]
        cmaster_image.load_full_info()
        cmaster_image.load_slice_info()

        for ifgs_dir in ifgs:
            ifg_dir = os.path.join(self.datastack_folder, ifgs_dir)

            master = ifgs_dir[-17:-9]
            slave = ifgs_dir[-8:]

            try:
                image_id = self.image_dates.index(master)
                master_image = self.images[self.image_dates[image_id]]
            except ValueError:
                master_image = []

            try:
                image_id = self.image_dates.index(slave)
                slave_image = self.images[self.image_dates[image_id]]
            except ValueError:
                slave_image = []

            self.interferograms[ifgs_dir[-17:]] = Interferogram(ifg_dir, master_image, slave_image, cmaster_image, self.master_slice_names)

            if master not in self.ifg_dates:
                self.ifg_dates.append(master)
            if slave not in self.ifg_dates:
                self.ifg_dates.append(slave)

        # combine the ifg and image dates
        self.dates = sorted(set(self.ifg_dates) - set(self.image_dates))

    def __call__(self, step, settings, coor, meta_type, file_type='', cores=6, memory=500, parallel=True):

        # If not defined first check whether we are dealing with a slave, cmaster or ifg step.
        # Should be defined by the user! But to warn in the case the likely wrong function is chosen

        # Run either all the slave, the coreg master or the ifg's
        if meta_type == 'cmaster':
            self.images[self.master_date](step, settings, coor, file_type, cmaster='',
                                          memory=memory, cores=cores, parallel=parallel)
        elif meta_type == 'slave':
            for key in self.images.keys():
                self.images[key](step, settings, coor, file_type, cmaster=self.images[self.master_date],
                                 memory=memory, cores=cores, parallel=parallel)

        elif meta_type == 'ifg':
            for key in self.interferograms.keys():
                master_key = key[:8]
                slave_key = key[9:]

                if master_key in self.images.keys():
                    master = self.images[master_key]
                else:
                    master = ''
                if slave_key in self.images.keys():
                    slave = self.images[slave_key]
                else:
                    slave = ''

                self.interferograms[key](step, settings, coor, file_type, slave=slave, master=master,
                                         cmaster=self.images[self.master_date], memory=memory, cores=cores, parallel=parallel)

    def export_to_geotiff(self, step, file_type, interferogram=True, slice=False):
        # Export files as geotiff for full stack.

        if isinstance(step, str):
            step = [step]
        if isinstance(file_type, str):
            file_type = [file_type]

        if interferogram:
            for im_key in list(self.interferograms.keys()):

                if slice:
                    for slice_key in list(self.interferograms[im_key].slices.keys()):
                        self.interferograms[im_key].slices[slice_key].image_create_geotiff(step, file_type)
                else:
                    self.interferograms[im_key].res_data.image_create_geotiff(step, file_type)
        else:
            for im_key in list(self.images.keys()):

                if slice:
                    for slice_key in list(self.images[im_key].slices.keys()):
                        self.images[im_key].slices[slice_key].image_create_geotiff(step, file_type)
                else:
                    self.images[im_key].res_data.image_create_geotiff(step, file_type)

    def add_master_res_info(self):
        # This function adds the .res information specific for the master image. For this image both resampline and the
        # topography/earth phase are not needed as it is the reference itself. Therefore these steps are added to the
        # .res file but simply referenced to the original data crop.

        # Note that this could cause a problem if multilooking of the original data is applied before creating the ifg.
        # However, this is not advised in almost all cases

        coor = CoordinateSystem()
        coor.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])

        for slice in self.images[self.master_date].slices.keys():

            coor.add_res_info(self.images[self.master_date].slices[slice])
            for step in ['reramp', 'earth_topo_phase']:
                if not self.images[self.master_date].slices[slice].process_control[step] == '1':
                    res_info = coor.create_meta_data([step], ['complex_int'])
                    res_info[step + '_output_file'] = 'crop.raw'
                    self.images[self.master_date].slices[slice].image_add_processing_step(step, res_info)

            # Create a file with zeros for both baseline, coreg, height_to_phase.
            filename = os.path.join(os.path.dirname(self.images[self.master_date].slices[slice].res_path), 'height_to_phase.raw')
            new_file = np.memmap(filename, dtype=np.float32, mode='w+', shape=tuple(coor.shape))
            new_file.flush()

            for step, file_types, file_data_types in zip(['baseline', 'height_to_phase'], [
                ['perpendicular_baseline', 'parallel_baseline', 'vertical_baseline', 'angle_baseline', 'total_baseline'],
                ['height_to_phase']], [['real4', 'real4', 'real4', 'real4', 'real4', 'real4'], ['real4']]):

                if not self.images[self.master_date].slices[slice].process_control[step] == '1':

                    res_info = coor.create_meta_data(file_types, file_data_types)
                    for file_type in file_types:
                        res_info[file_type + '_output_file'] = 'height_to_phase.raw'
                    self.images[self.master_date].slices[slice].image_add_processing_step(step, res_info)

            self.images[self.master_date].slices[slice].write()

    def create_network_ifgs(self, network_type='temp_baseline', temp_baseline=14, n_images=1):

        # Get all the dates in the stack.
        date_int = np.sort([int(key) for key in self.images.keys()])
        master_int = int(self.master_date)
        dates = np.array([datetime.datetime.strptime(str(date), '%Y%m%d') for date in date_int])

        ifg_pairs = []

        if network_type == 'temp_baseline':
            days = np.array([diff.days for diff in dates - np.min(dates)])

            # Define network based on network type
            for n in np.arange(len(days)):
                ids = np.where((days - days[n] > 0) * (days - days[n] <= temp_baseline))[0]
                for id in ids:
                    ifg_pairs.append([n, id])

        elif network_type == 'daisy_chain':
            n_im = len(dates)
            for i in np.arange(n_im):
                for n in np.arange(0, n_images + 1):
                    if n + i < n_im:
                        ifg_pairs.append([i, n])

        elif network_type == 'single_master':

            master_n = np.where(date_int == master_int)[0][0]

            for n in np.arange(len(date_int)):
                if n != master_n:
                    ifg_pairs.append([master_n, n])

        # Finally create the requested ifg if they do not exist already
        ifg_ids = self.interferograms.keys()
        cmaster_key = self.master_date

        for ifg_pair in ifg_pairs:

            master_key = str(date_int[ifg_pair[0]])
            slave_key = str(date_int[ifg_pair[1]])

            ifg_key_1 = master_key + '_' + slave_key
            ifg_key_2 = slave_key + '_' + master_key

            if not ifg_key_1 in ifg_ids and not ifg_key_2 in ifg_ids:
                ifg = Interferogram(slave=self.images[slave_key], master=self.images[master_key], cmaster=self.images[cmaster_key])
                self.interferograms[ifg_key_1] = ifg

    def create_SRTM_input_data(self, srtm_folder, username, password, buf=0.2, rounding=0.2,  srtm_type='SRTM3'):
        # This creates the input DEM for the different slices of the master image.

        self.download_SRTM_dem(srtm_folder, username, password, buf, rounding, srtm_type)

        # First full image
        image = self.images[self.master_date].res_data
        dem_path = os.path.join(os.path.dirname(image.res_path), 'DEM_WGS84_stp_' + srtm_type[-1] + '_' + srtm_type[-1] + '.raw')

        if not self.images[self.master_date].res_data.process_control['import_DEM'] == '1' or not \
                os.path.exists(dem_path):
            SRTM_dat = CreateSrtmDem(image, srtm_folder, buf=buf, rounding=rounding, srtm_type=srtm_type)
            SRTM_dat()
            image.write()

        # Then slices
        for key in self.images[self.master_date].slices.keys():
            slice = self.images[self.master_date].slices[key]
            dem_path = os.path.join(os.path.dirname(slice.res_path), 'DEM_WGS84_stp_' + srtm_type[-1] + '_' + srtm_type[-1] + '.raw')

            if not self.images[self.master_date].slices[key].process_control['import_DEM'] == '1' or not \
                    os.path.exists(dem_path):
                SRTM_dat = CreateSrtmDem(slice, srtm_folder, buf=buf, rounding=rounding, srtm_type=srtm_type)
                SRTM_dat()
                slice.write()

    def download_SRTM_dem(self, srtm_folder, username, password, buf=0.5, rounding=0.5, srtm_type='SRTM3'):
        # Downloads the needed srtm data for this datastack. srtm_folder is the folder the downloaded srtm tiles are
        # stored.
        # Username and password can be obtained at https://lpdaac.usgs.gov
        # Documentation: https://lpdaac.usgs.gov/sites/default/files/public/measures/docs/NASA_SRTM_V3.pdf

        # Description srtm data: https://lpdaac.usgs.gov/dataset_discovery/measures/measures_products_table/SRTMGL1_v003
        # Description srtm q data: https://lpdaac.usgs.gov/node/505

        download = SrtmDownload(srtm_folder, username, password, srtm_type)
        download(self.images[self.master_date].res_data, buf=buf, rounding=rounding)

    # Possibly an extension to use of free tandem-x data?
