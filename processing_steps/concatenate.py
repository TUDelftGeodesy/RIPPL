# This class uses a shapefile and a radar image to detect which burst are part of the image and which bursts are not
# The function has two different options:
# 1. Combine an input image and a shapefile to find the corresponding bursts.
# 2. Combine information on the specified bursts and a new image to check whether all bursts are available in this
#       image and how they are linked.


# The following class creates an interferogram from a master and slave image.

from doris_processing.image_data import ImageData
from doris_processing.processing_steps.interfero import Interfero
from collections import OrderedDict, defaultdict
import datetime
import numpy as np
import os
import copy
import logging


class Concatenate(object):

    """
    :type s_pix = int
    :type s_lin = int
    :type shape = list
    """

    def __init__(self, slices_meta, concat_meta='', multilook='', oversampling='', offset_image='', offset_burst=''):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        self.slices = []
        meta_type = slices_meta[0].res_type

        if not isinstance(slices_meta, list):
            print('slices meta should contain a list with meta data files')
        for slice in slices_meta:
            if isinstance(slice, str):
                if len(slice) != 0:
                    self.slices.append(ImageData(slice, meta_type))
            elif isinstance(slice, ImageData):
                self.slices.append(slice)

        self.ml_str, self.multilook, self.oversampling, self.offset_image = Interfero.get_ifg_str(multilook, oversampling, offset_image)

        self.az_min_coor = []
        self.ra_min_coor = []
        self.az_max_coor = []
        self.ra_max_coor = []
        self.d_type = []
        self.d_type_str = []

        self.az_offset = []
        self.ra_offset = []
        self.az_new_coor = []
        self.ra_new_coor = []
        self.offsets = []
        self.coors = []

        self.ml_slices = []
        if offset_burst == '':
            offset_burst = [20, 100]
        self.offset_burst = offset_burst

        self.find_slice_multilook_offset(self.multilook, self.oversampling, self.offset_burst)

        if isinstance(concat_meta, str):
            if len(concat_meta) != 0:
                self.concat = ImageData(concat_meta, meta_type)
            else:
                self.create_concat_meta(meta_type)

        elif isinstance(concat_meta, ImageData):
            self.concat = concat_meta

    def __call__(self, step='interferogram', data_type='Data', in_data='memory'):

        self.load_slices(step, in_data, data_type=data_type)
        if len(self.ml_slices) == 0:
            print('Missing input data for concatenation of image ' + self.concat.folder + '. Aborting..')
            return False

        try:
            self.add_meta_data(step, data_type)

            dat_type = data_type + self.ml_str
            cutoff = np.array(self.offset_image) / np.array(self.multilook)
            if cutoff[0] * self.multilook[0] != self.offset_burst[0]:
                cutoff[0] += 1
            if cutoff[1] * self.multilook[1] != self.offset_burst[1]:
                cutoff[1] += 1

            if in_data == 'disk':
                self.concat.image_create_disk(step, dat_type)

                for slice, c in zip(self.ml_slices, self.coors):
                    s = slice.shape
                    self.concat.data_disk[step][dat_type][c[0]:c[0] + s[0] - cutoff[0], c[1]:c[1] + s[1] - cutoff[1]] = \
                        slice[:-cutoff[0], :-cutoff[1]]

            if in_data == 'memory':
                shape = self.concat.data_sizes[step][dat_type]
                offset = self.concat.data_offset[step][dat_type]
                self.concat.image_new_data_memory(np.zeros(shape).astype(self.d_type), step,
                                                  s_lin=offset[0], s_pix=offset[1], file_type=dat_type)

                for slice, c in zip(self.ml_slices, self.coors):
                    s = slice.shape
                    self.concat.data_memory[step][dat_type][c[0]:c[0] + s[0] - cutoff[0], c[1]:c[1] + s[1] - cutoff[1]] = \
                        slice[:-cutoff[0], :-cutoff[1]]

            return True

        except Exception:
            log_file = os.path.join(self.concat.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed processing azimuth_elevation_angle for ' + self.concat.folder + '. Check ' + log_file + ' for details.')
            print('Failed processing azimuth_elevation_angle for ' + self.concat.folder + '. Check ' + log_file + ' for details.')

            return False

    def create_concat_meta(self, meta_type):
        # This code create a new concatenated slice.

        # First define the new folder and .res name
        filename = os.path.join(os.path.dirname(os.path.dirname(self.slices[0].res_path)), 'info.res')

        # Then create a single or interferogram file
        meta = ImageData('', meta_type)
        meta.res_path = filename
        meta.folder = os.path.dirname(self.slices[0].folder)

        lat_lim = [np.min(np.array([slice.lat_lim[0] for slice in self.slices])),
                   np.max(np.array([slice.lat_lim[1] for slice in self.slices]))]
        lon_lim = [np.min(np.array([slice.lon_lim[0] for slice in self.slices])),
                   np.max(np.array([slice.lon_lim[1] for slice in self.slices]))]

        # Add the readfiles and orbit information
        if self.slices[0].process_control['coreg_readfiles'] == '1':
            if self.slices[0].res_type == 'single':
                steps_meta = ['', 'coreg_']
            elif self.slices[0].res_type == 'interferogram':
                steps_meta = ['coreg_']
        else:
            steps_meta = ['']

        for step_meta in steps_meta:
            meta.image_add_processing_step(step_meta + 'readfiles', copy.copy(self.slices[0].processes[step_meta + 'readfiles']))
            meta.image_add_processing_step(step_meta + 'orbits', copy.copy(self.slices[0].processes[step_meta + 'orbits']))
            meta.image_add_processing_step(step_meta + 'crop',copy.copy(self.slices[0].processes[step_meta + 'crop']))

            # Adapt the readfiles and orbit information
            az_date = datetime.datetime.strptime(self.slices[0].processes[step_meta + 'readfiles']
                                                 ['First_pixel_azimuth_time (UTC)'][:10], '%Y-%m-%d')
            az_time = (az_date + datetime.timedelta(seconds=self.min_az)).strftime('%Y-%m-%dT%H:%M:%S.%f')
            ra_time = str(self.min_ra * 1000)
            az_max = np.max(self.az_max_coor)
            ra_max = np.max(self.ra_max_coor)

            # Adapt the crop information
            meta.processes[step_meta + 'crop']['Data_output_file'] = 'crop.raw'
            meta.processes[step_meta + 'crop']['Data_output_format'] = 'complex_int'
            meta.processes[step_meta + 'crop']['Data_first_line'] = '1'
            meta.processes[step_meta + 'crop']['Data_first_pixel'] = '1'
            meta.processes[step_meta + 'crop']['Data_pixels'] = str(int(ra_max))
            meta.processes[step_meta + 'crop']['Data_lines'] = str(int(az_max))
            meta.processes[step_meta + 'crop'].pop('Data_first_line (w.r.t. tiff_image)')
            meta.processes[step_meta + 'crop'].pop('Data_last_line (w.r.t. tiff_image)')

            # Add coordinates and azimuth/range timing
            meta.processes[step_meta + 'readfiles']['Number_of_lines_original'] = str(int(az_max))
            meta.processes[step_meta + 'readfiles']['Number_of_pixels_original'] = str(int(ra_max))
            meta.processes[step_meta + 'readfiles']['First_pixel_azimuth_time (UTC)'] = az_time
            meta.processes[step_meta + 'readfiles']['Range_time_to_first_pixel (2way) (ms)'] = ra_time

            # Remove all burst specific information to avoid confusion
            meta.processes[step_meta + 'readfiles'].pop('Number_of_lines_Swath')
            meta.processes[step_meta + 'readfiles'].pop('SWATH')
            meta.processes[step_meta + 'readfiles'].pop('number_of_pixels_Swath')
            meta.processes[step_meta + 'readfiles'].pop('total_Burst')
            meta.processes[step_meta + 'readfiles'].pop('Burst_number_index')

            # Add the new coordinates. This is only an approximation using a combination of the burst lat/lon coordinates
            meta.processes[step_meta + 'readfiles']['Scene_ul_corner_latitude'] = str(lat_lim[1])
            meta.processes[step_meta + 'readfiles']['Scene_ur_corner_latitude'] = str(lat_lim[1])
            meta.processes[step_meta + 'readfiles']['Scene_lr_corner_latitude'] = str(lat_lim[0])
            meta.processes[step_meta + 'readfiles']['Scene_ll_corner_latitude'] = str(lat_lim[0])
            meta.processes[step_meta + 'readfiles']['Scene_ul_corner_longitude'] = str(lon_lim[0])
            meta.processes[step_meta + 'readfiles']['Scene_ur_corner_longitude'] = str(lon_lim[1])
            meta.processes[step_meta + 'readfiles']['Scene_lr_corner_longitude'] = str(lon_lim[1])
            meta.processes[step_meta + 'readfiles']['Scene_ll_corner_longitude'] = str(lon_lim[0])

            meta.processes[step_meta + 'readfiles'].pop('First_pixel (w.r.t. tiff_image)')
            meta.processes[step_meta + 'readfiles'].pop('First_line (w.r.t. tiff_image)')
            meta.processes[step_meta + 'readfiles'].pop('Last_pixel (w.r.t. tiff_image)')
            meta.processes[step_meta + 'readfiles'].pop('Last_line (w.r.t. tiff_image)')
            meta.processes[step_meta + 'readfiles'].pop('Number_of_lines_burst')
            meta.processes[step_meta + 'readfiles'].pop('Number_of_pixels_burst')

            meta.processes[step_meta + 'readfiles'].pop('DC_reference_azimuth_time')
            meta.processes[step_meta + 'readfiles'].pop('DC_reference_range_time')
            meta.processes[step_meta + 'readfiles'].pop('Xtrack_f_DC_constant (Hz, early edge)')
            meta.processes[step_meta + 'readfiles'].pop('Xtrack_f_DC_linear (Hz/s, early edge)')
            meta.processes[step_meta + 'readfiles'].pop('Xtrack_f_DC_quadratic (Hz/s/s, early edge)')

            meta.processes[step_meta + 'readfiles'].pop('FM_reference_azimuth_time')
            meta.processes[step_meta + 'readfiles'].pop('FM_reference_range_time')
            meta.processes[step_meta + 'readfiles'].pop('FM_polynomial_constant_coeff (Hz, early edge)')
            meta.processes[step_meta + 'readfiles'].pop('FM_polynomial_linear_coeff (Hz/s, early edge)')
            meta.processes[step_meta + 'readfiles'].pop('FM_polynomial_quadratic_coeff (Hz/s/s, early edge)')

            meta.processes[step_meta + 'readfiles'].pop('Datafile')
            meta.processes[step_meta + 'readfiles'].pop('Dataformat')

        self.concat = meta

    def add_meta_data(self, step, data_type):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.

        if step in self.concat.processes.keys():
            meta_info = self.concat.processes[step]
        else:
            meta_info = OrderedDict()

        # Add the readfiles and orbit information
        if self.slices[0].process_control['coreg_readfiles'] == '1':
            step_meta = 'coreg_'
        else:
            step_meta = ''

        dat = data_type + self.ml_str

        meta_info[dat + '_output_file'] = step + self.ml_str + '.raw'
        meta_info[dat + '_output_format'] = self.d_type_str

        new_lines = np.max(self.az_max_coor) * self.oversampling[0] / self.multilook[0]
        new_pixels = np.max(self.ra_max_coor) * self.oversampling[1] / self.multilook[1]
        meta_info[dat + '_lines'] = str(new_lines)
        meta_info[dat + '_pixels'] = str(new_pixels)
        meta_info[dat + '_multilook_azimuth'] = str(self.multilook[0])
        meta_info[dat + '_multilook_range'] = str(self.multilook[1])
        meta_info[dat + '_oversampling_azimuth'] = str(self.oversampling[0])
        meta_info[dat + '_oversampling_range'] = str(self.oversampling[1])
        meta_info[dat + '_offset_azimuth'] = str(self.offset_image[0])
        meta_info[dat + '_offset_range'] = str(self.offset_image[1])
        meta_info[dat + '_first_line'] = str(1)
        meta_info[dat + '_first_pixel'] = str(1)

        self.concat.image_add_processing_step(step, meta_info)

    @staticmethod
    def processing_info():
        # Information on this processing step
        input_dat = defaultdict()
        input_dat['master']['crop', 'earth_dem_phase'] = ['Data']
        input_dat['slave']['crop', 'earth_dem_phase'] = ['Data']

        output_dat = dict()
        output_dat['ifgs']['interferogram'] = ['Data']

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 3

        return input_dat, output_dat, mem_use

    def find_slice_coors(self):
        # This function is used to create an oversight of the coordinates of the available slices. This results in an
        # oversight of the start pixel/line of every burst compared to the full image.

        if self.slices[0].process_control['coreg_readfiles'] == '1':
            step_meta = 'coreg_readfiles'
            step_crop = 'coreg_crop'
        else:
            step_meta = 'readfiles'
            step_crop = 'crop'

        # First find the first pixel azimuth and range pixels
        az_times = []
        ra_times = []
        first_line = []
        last_line = []
        first_pixel = []
        last_pixel = []

        for slice in self.slices:
            az_time = slice.processes[step_meta]['First_pixel_azimuth_time (UTC)']
            az_seconds = (datetime.datetime.strptime(az_time, '%Y-%m-%dT%H:%M:%S.%f') -
                          datetime.datetime.strptime(az_time[:10], '%Y-%m-%d'))
            az_times.append(az_seconds.seconds + az_seconds.microseconds / 1000000.0)
            ra_times.append(float(slice.processes[step_meta]['Range_time_to_first_pixel (2way) (ms)']) / 1000)
            az_step = float(slice.processes[step_meta]['Azimuth_time_interval (s)'])
            ra_step = 1 / float(slice.processes[step_meta]['Range_sampling_rate (computed, MHz)']) / 1000000
            first_lin = int(slice.processes[step_crop]['Data_first_line']) - 1
            first_line.append(first_lin)
            first_pix = int(slice.processes[step_crop]['Data_first_pixel']) - 1
            first_pixel.append(first_pix)
            last_line.append(first_lin + int(slice.processes[step_crop]['Data_lines']))
            last_pixel.append(first_pix + int(slice.processes[step_crop]['Data_pixels']))

        first_line = np.array(first_line)
        first_pixel = np.array(first_pixel)
        last_line = np.array(last_line)
        last_pixel = np.array(last_pixel)
        az_times = np.array(az_times)
        ra_times = np.array(ra_times)

        # Then use the extend with respect to the master / slave image to get the maximum and minimum range/azimuth times
        # Find the lowest range/azimuth times and use that as pixel 1,1
        self.min_az = np.min(az_times + first_line * az_step)
        self.min_ra = np.min(ra_times + first_pixel * ra_step)

        # Create a table of line and pixel coordinates according to this reference
        self.az_min_coor = ((az_times - self.min_az) / az_step + first_line).astype(np.int32)
        self.ra_min_coor = ((ra_times - self.min_ra) / ra_step + first_pixel).astype(np.int32)
        self.az_max_coor = ((az_times - self.min_az) / az_step + last_line).astype(np.int32)
        self.ra_max_coor = ((ra_times - self.min_ra) / ra_step + last_pixel).astype(np.int32)

    def find_slice_multilook_offset(self, multilook='', oversampling='', offset=''):
        # To concatenate the full image after multilooking we will have to know where to start our multilooking window
        # for every burst. This is calculated in this function. The border offset defines how far we should be from
        # the borders of the image, to avoid including empty pixels. Generally an offset of 20 pixels for the azimuth and
        # 200 pixels for range will suffice.

        sample, self.multilook, self.oversampling, offset = Interfero.get_ifg_str(multilook, oversampling, offset)

        if len(self.az_min_coor) == 0:
            self.find_slice_coors()

        self.az_offset = []
        self.ra_offset = []
        self.az_new_coor = []
        self.ra_new_coor = []
        self.offsets = []
        self.coors = []

        az = self.multilook[0] / self.oversampling[0]
        ra = self.multilook[1] / self.oversampling[1]
        ovr_lin = float(self.multilook[0] / 2.0) - (float(self.multilook[0]) / float(self.oversampling[0]) / 2)
        ovr_pix = float(self.multilook[1] / 2.0) - (float(self.multilook[1]) / float(self.oversampling[1]) / 2)

        # Extend offset to maintain needed buffer.
        if ovr_lin - float(int(ovr_lin)) > 0.1 or ovr_pix - float(int(ovr_pix)) > 0.1:
            print('Warning. Averaging window for oversampling shifted half a pixel in azimuth or range direction.')
            print('To prevent make sure that oversampling value is odd or multilook/oversampling is even')
            ovr_lin = int(np.ceil(ovr_lin))
            ovr_pix = int(np.ceil(ovr_pix))
        else:
            ovr_lin = int(ovr_lin)
            ovr_pix = int(ovr_pix)

        # Extend offset.
        self.offset = np.array(offset) + np.array([ovr_lin, ovr_pix])

        # Use the table from the find slice coors function.
        for az_coor, ra_coor in zip(self.az_min_coor, self.ra_min_coor):
            az_rest = int(az_coor + self.offset[0]) % az
            az_new_coor = int(az_coor + self.offset[0]) / az
            if az_rest != 0:
                az_new_coor += 1
            if az_new_coor < 0:  # In case we are at the very border of the picture.
                az_new_coor = 0
            az_offset = (az_new_coor * az) - az_coor

            ra_rest = int(ra_coor + self.offset[1]) % ra
            ra_new_coor = int(ra_coor + self.offset[1]) / ra
            if ra_rest != 0:
                ra_new_coor += 1
            if ra_new_coor < 0:  # In case we are at the very border of the picture.
                ra_new_coor = 0
            ra_offset = (ra_new_coor * ra) - ra_coor
            
            self.az_offset.append(az_offset)
            self.ra_offset.append(ra_offset)
            self.az_new_coor.append(az_new_coor)
            self.ra_new_coor.append(ra_new_coor)
            self.offsets.append([az_offset, ra_offset])
            self.coors.append([az_new_coor, ra_new_coor])

    def load_slices(self, step, in_data='memory', data_type='Data', multilook='', oversampling='', offset=''):
        # This function checks whether the needed multilooked data is available and throws an error if not.

        if len(self.az_offset) == 0:
            self.find_slice_multilook_offset(multilook, offset, oversampling)

        self.ml_slices = []
        self.d_type = []

        if self.multilook == [1, 1]:
            int_str = ''
        else:
            int_str = 'ml_' + str(self.multilook[0]) + '_' + str(self.multilook[1])
        if self.oversampling == [1, 1]:
            ovr_str = ''
        else:
            ovr_str = 'ovr_' + str(self.oversampling[0]) + '_' + str(self.oversampling[1])

        for slice, az_offset, ra_offset in zip(self.slices, self.az_offset, self.ra_offset):

            offset = [az_offset, ra_offset]
            if offset == [0, 0]:
                buf_str = ''
            else:
                buf_str = 'off_' + str(offset[0]) + '_' + str(offset[1])

            sample = ''
            if int_str:
                sample = sample + '_' + int_str
            if buf_str:
                sample = sample + '_' + buf_str
            if ovr_str:
                sample = sample + '_' + ovr_str

            type_dat = data_type + sample

            if in_data == 'disk' and step in slice.data_disk.keys():
                if type_dat in slice.data_disk[step].keys():
                    self.ml_slices.append(slice.data_disk[step][type_dat])
                    self.d_type_str = slice.data_types[step][type_dat]
                    self.d_type = slice.dtype_numpy[slice.data_types[step][type_dat]]
                else:
                    print('One of the needed slices is missing for concatenation')
            elif in_data == 'memory' and step in slice.data_memory.keys():
                if type_dat in slice.data_memory[step].keys():
                    self.ml_slices.append(slice.data_memory[step][type_dat])
                    self.d_type_str = slice.data_types[step][type_dat]
                    self.d_type = slice.dtype_numpy[slice.data_types[step][type_dat]]
                else:
                    print('One of the needed slices is missing for concatenation')
            else:
                print('One of the needed slices is missing for concatenation')
