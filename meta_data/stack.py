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
import datetime
import numpy as np

from rippl.meta_data.slc import SLC
from rippl.meta_data.interferogram import Interferogram
from rippl.external_DEMs.srtm.srtm_download import SrtmDownload
from rippl.meta_data.interferogram_network import InterferogramNetwork


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
        image_dirs = sorted([os.path.join(self.datastack_folder, x) for x in dirs if (len(x) == 8 and
                         start <= int(x) <= end) or x == self.master_date])
        ifg_dirs = sorted([os.path.join(self.datastack_folder, x) for x in dirs if len(x) == 17 and
                       start <= int(x[:8]) <= end and start <= int(x[9:]) <= end])

        # Load individual images.
        for image_dir in image_dirs:
            if image_dir not in self.image_dates:
                self.images[os.path.dirname(image_dir)] = SLC(image_dir, slice_list=self.master_slice_names)
                self.image_dates.append(os.path.basename(image_dir))

        # Load master date information.
        cmaster_image = self.images[self.master_date]
        cmaster_image.load_full_meta()
        cmaster_image.load_slice_meta()

        # Load ifgs
        for ifg_dir in ifg_dirs:
            if ifg_dir not in self.ifg_dates:
                self.interferograms[ifg_dir[-17:]] = Interferogram(ifg_dir, slice_list=self.master_slice_names)
                self.ifg_dates.append(os.path.basename(ifg_dir))

        # combine the ifg and image dates
        self.dates = sorted(set(self.ifg_dates) - set(self.image_dates))

    def create_interferogram_network(self, image_baselines=[], network_type='temp_baseline',
                                     temporal_baseline=60, temporal_no=3, spatial_baseline=2000):
        # This method will call the create interferogram network class.
        # Run after reading in the datastack.

        network = InterferogramNetwork(self.images.keys(), self.master_date, image_baselines, network_type,
                                       temporal_baseline, temporal_no, spatial_baseline)
        ifg_pairs = network.ifg_pairs

        # Finally create the requested ifg if they do not exist already
        ifg_ids = self.interferograms.keys()
        cmaster_key = self.master_date
        date_int = np.sort([int(key) for key in self.images.keys()])

        for ifg_pair in ifg_pairs:

            master_key = str(date_int[ifg_pair[0]])
            slave_key = str(date_int[ifg_pair[1]])

            ifg_key_1 = master_key + '_' + slave_key
            ifg_key_2 = slave_key + '_' + master_key

            if not ifg_key_1 in ifg_ids and not ifg_key_2 in ifg_ids:
                ifg = Interferogram(slave=self.images[slave_key], master=self.images[master_key],
                                    cmaster=self.images[cmaster_key])
                self.interferograms[ifg_key_1] = ifg

    def stack_data_iterator(self, coordinates, process, process_type='',
                            images=True, interferograms=True,  full_image=True, slices=False):
        # Get all the full-images or slices




    def download_SRTM_dem(self, srtm_folder, username, password, buf=0.5, rounding=0.5, srtm_type='SRTM3', parallel=True):
        # Downloads the needed srtm data for this datastack. srtm_folder is the folder the downloaded srtm tiles are
        # stored.
        # Username and password can be obtained at https://lpdaac.usgs.gov
        # Documentation: https://lpdaac.usgs.gov/sites/default/files/public/measures/docs/NASA_SRTM_V3.pdf

        # Description srtm data: https://lpdaac.usgs.gov/dataset_discovery/measures/measures_products_table/SRTMGL1_v003
        # Description srtm q data: https://lpdaac.usgs.gov/node/505

        download = SrtmDownload(srtm_folder, username, password, srtm_type)
        download(self.images[self.master_date].data, buf=buf, rounding=rounding, parallel=parallel)

    def download_ECMWF_data(self, dat_type, ecmwf_data_folder, latlim='', lonlim= '', processes=6, parallel=True):
        # Download ECMWF data for whole dataset at once. This makes this process much faster.

        # Check the progress of your download at:
        # 1 https://apps.ecmwf.int/webmars/joblist/ (for operational products)
        # 2 https://cds.climate.copernicus.eu/cdsapp#!/yourrequests (for ERA5 data)

        try:
            from rippl.NWP_simulations.ECMWF.ecmwf_download import ECMWFdownload
        except:
            print('One of the ecmwf or grib reading packages is not configured correctly')
            return

        if len(latlim) != 2:
            latlim = [45, 56]
        if len(lonlim) != 2:
            lonlim = [-2, 12]

        down_dates = [datetime.datetime.strptime(d, '%Y%m%d') for d in self.images.keys()]

        download = ECMWFdownload(latlim, lonlim, ecmwf_data_folder, dat_type, processes, parallel=parallel)
        download.prepare_download(down_dates)
        download.download()
