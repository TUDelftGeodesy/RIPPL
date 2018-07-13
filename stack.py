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
import datetime


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
        self.master_slices = []
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

            self.master_slices.append('NoData')
        l.close()

    def read_stack(self):
        # This function reads the whole stack in memory. A stack consists of:
        # - images > with individual slices (yyyymmdd)
        # - interferograms > with individual slices if needed. (yyyymmdd_yyyymmdd)

        dirs = next(os.walk(self.datastack_folder))[1]
        images = sorted([os.path.join(self.datastack_folder, x) for x in dirs if len(x) == 8])
        ifgs = sorted([os.path.join(self.datastack_folder, x) for x in dirs if len(x) == 17])

        for im_dir in images:
            image_dir = os.path.join(self.datastack_folder, im_dir)

            if im_dir not in self.image_dates:
                self.images[im_dir[-8:]] = Image(image_dir, self.master_slice_names)
                self.image_dates.append(im_dir[-8:])

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

            self.interferograms[ifgs_dir[-17:]] = Interferogram(ifg_dir, master_image, slave_image, self.master_slice_names)

            if master not in self.ifg_dates:
                self.ifg_dates.append(master)
            if slave not in self.ifg_dates:
                self.ifg_dates.append(slave)

        # combine the ifg and image dates
        self.dates = sorted(set(self.ifg_dates) - set(self.image_dates))
