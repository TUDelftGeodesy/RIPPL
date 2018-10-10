# This class uses a shapefile and a radar image to detect which burst are part of the image and which bursts are not
# The function has two different options:
# 1. Combine an input image and a shapefile to find the corresponding bursts.
# 2. Combine information on the specified bursts and a new image to check whether all bursts are available in this
#       image and how they are linked.


# The following class creates an interferogram from a master and slave image.

from image_data import ImageData
from find_coordinates import FindCoordinates
from coordinate_system import CoordinateSystem
from collections import OrderedDict, defaultdict
import datetime
import numpy as np
import os
import copy
import logging


class Concatenate(object):

    def __init__(self, meta_slices, coordinates, meta='', step='interferogram', file_type=''):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        self.step = step
        if file_type == '':
            self.file_type = step
        else:
            self.file_type = file_type

        self.meta_slices = []
        for slice in meta_slices:
            if isinstance(slice, ImageData):
                self.meta_slices.append(slice)
            else:
                return

        self.data_type = self.meta_slices[0].data_types[step][file_type]

        if meta == '':
            self.meta = Concatenate.create_concat_meta(meta_slices)
        elif isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        self.coordinates = coordinates
        dummy, self.coordinates_slices = Concatenate.find_slice_coordinates(self.meta_slices, self.coordinates)

    def __call__(self, out_data='memory'):

        if len(self.meta_slices) == 0:
            print('Missing input data for concatenation of image ' + self.meta.folder + '. Aborting..')
            return False

        try:

            Concatenate.add_meta_data(self.meta, self.coordinates, self.file_type, self.data_type)

            if out_data == 'disk':
                self.meta.image_create_disk(self.step, self.file_type)
                data = self.meta.data_disk[self.step][self.file_type]
            elif out_data == 'memory':
                empty_image = np.zeros(self.coordinates.shape).astype(self.meta.dtype_numpy[self.data_type])
                self.meta.image_new_data_memory(empty_image, self.step, 0, 0, self.file_type)
                data = self.meta.data_memory[self.step][self.file_type]
            else:
                print('out_data should either be disk or memory')
                return

            # Load the data from the slices
            data_slices = Concatenate.load_slices(self.step, self.meta_slices, self.coordinates_slices, self.file_type)

            for data_slice, coordinates_slice in zip(data_slices, self.coordinates_slices):

                s_pix = coordinates_slice.first_pixel - 1
                s_lin = coordinates_slice.first_line - 1
                e_pix = coordinates_slice.first_pixel + coordinates_slice.shape[1] - 1
                e_lin = coordinates_slice.first_line + coordinates_slice.shape[0] - 1

                # Use different methods with different type of data. Complex data is added (radar data), while all other
                # data is simply replaced (information on geocoding or otherwise)
                if self.data_type in ['complex_int', 'complex_short', 'complex_real4', 'tiff'] or \
                        self.step in ['square_amplitude']:
                    data[s_lin:e_lin, s_pix:e_pix] += data_slice
                else:
                    data[s_lin:e_lin, s_pix:e_pix] = data_slice

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed processing azimuth_elevation_angle for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed processing azimuth_elevation_angle for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def add_meta_data(meta, coordinates, step, file_type='', data_type=''):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if step in meta.processes.keys():
            meta_info = meta.processes[step]
        else:
            meta_info = OrderedDict()

        if file_type == '':
            file_type = step

        meta_info = coordinates.create_meta_data([file_type], [data_type], meta_info)
        meta.image_add_processing_step(step, meta_info)

    @staticmethod
    def processing_info(meta_slices, coordinates, step, file_type=''):
        # This is a special case where we go from slices to full images. Therefore we need some extra info to find
        # the input/output information. This function needs the input slices including the information of the radar
        # coordinates of the original or coreg grid (depending on what is needed.)

        # For multiprocessing it is important that coreg master grid is used.
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        meta, coordinates_slices = Concatenate.find_slice_coordinates(meta_slices, coordinates)
        slice_names = [os.path.basename(slice.folder) for slice in meta_slices]
        slice_file_names = [file_type + coor.sample + '.raw' for coor in coordinates_slices]

        # Three input files needed x, y, z coordinates
        input_dat = defaultdict()
        input_dat['meta'][step][file_type]['files'] = slice_file_names
        input_dat['meta'][step][file_type]['slice_names'] = slice_names
        input_dat['meta'][step][file_type]['coordinates'] = coordinates_slices
        input_dat['meta'][step][file_type]['slice'] = 'True'

        # line and pixel output files.
        output_dat = defaultdict()
        output_dat['meta'][step][file_type]['file'] = [file_type + coordinates.sample + '.raw']
        output_dat['meta'][step][file_type]['coordinates'] = coordinates
        output_dat['meta'][step][file_type]['slice'] = 'False'

        # Data is used only once for concatenation.
        mem_use = 1

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, step, output_file_steps=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not output_file_steps:
            meta_info = meta.processes[step]
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            output_file_steps = [filename[:-13] for filename in output_file_keys]

        for s in output_file_steps:
            meta.image_create_disk(step, s)

    @staticmethod
    def save_to_disk(meta, step, output_file_steps=''):

        if not output_file_steps:
            meta_info = meta.processes[step]
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            output_file_steps = [filename[:-13] for filename in output_file_keys]

        for s in output_file_steps:
            meta.image_memory_to_disk(step, s)

    @staticmethod
    def create_concat_meta(meta_slices):
        # This code create a new concatenated slice.

        # First define the new folder and .res name
        filename = os.path.join(os.path.dirname(os.path.dirname(meta_slices[0].res_path)), 'info.res')

        # Then create a single or interferogram file
        meta_type = meta_slices[0].res_type
        meta = ImageData('', meta_type)
        meta.res_path = filename
        meta.folder = os.path.dirname(meta_slices[0].folder)

        lat_lim = [np.min(np.array([slice.lat_lim[0] for slice in meta_slices])),
                   np.max(np.array([slice.lat_lim[1] for slice in meta_slices]))]
        lon_lim = [np.min(np.array([slice.lon_lim[0] for slice in meta_slices])),
                   np.max(np.array([slice.lon_lim[1] for slice in meta_slices]))]

        min_az, min_ra, shape = Concatenate.find_radar_max_extend(meta_slices)

        # Add the readfiles and orbit information
        if meta_slices[0].process_control['coreg_readfiles'] == '1':
            if meta_slices[0].res_type == 'single':
                steps_meta = ['', 'coreg_']
            elif meta_slices[0].res_type == 'interferogram':
                steps_meta = ['coreg_']
            else:
                return
        else:
            steps_meta = ['']

        for step_meta in steps_meta:
            meta.image_add_processing_step(step_meta + 'readfiles', copy.copy(meta_slices[0].processes[step_meta + 'readfiles']))
            meta.image_add_processing_step(step_meta + 'orbits', copy.copy(meta_slices[0].processes[step_meta + 'orbits']))
            meta.image_add_processing_step(step_meta + 'crop',copy.copy(meta_slices[0].processes[step_meta + 'crop']))

            # Adapt the readfiles and orbit information
            az_date = datetime.datetime.strptime(meta_slices[0].processes[step_meta + 'readfiles']
                                                 ['First_pixel_azimuth_time (UTC)'][:10], '%Y-%m-%d')
            az_time = (az_date + datetime.timedelta(seconds=min_az)).strftime('%Y-%m-%dT%H:%M:%S.%f')
            ra_time = str(min_ra * 1000)

            # Adapt the crop information
            meta.processes[step_meta + 'crop']['Data_output_file'] = 'crop.raw'
            meta.processes[step_meta + 'crop']['Data_output_format'] = 'complex_int'
            meta.processes[step_meta + 'crop']['Data_first_line'] = '1'
            meta.processes[step_meta + 'crop']['Data_first_pixel'] = '1'
            meta.processes[step_meta + 'crop']['Data_pixels'] = str(int(shape[0]))
            meta.processes[step_meta + 'crop']['Data_lines'] = str(int(shape[1]))
            meta.processes[step_meta + 'crop'].pop('Data_first_line (w.r.t. tiff_image)')
            meta.processes[step_meta + 'crop'].pop('Data_last_line (w.r.t. tiff_image)')

            # Change from slice to full image
            meta.processes[step_meta + 'readfiles']['slice'] = 'False'

            # Add coordinates and azimuth/range timing
            meta.processes[step_meta + 'readfiles']['Number_of_lines_original'] = str(int(shape[0]))
            meta.processes[step_meta + 'readfiles']['Number_of_pixels_original'] = str(int(shape[0]))
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

        return meta

    @staticmethod
    def find_radar_max_extend(meta_slices):
        # Finds the maximum extend of the image in original line, pixel coordinates.

        if meta_slices[0].process_control['coreg_readfiles'] == '1':
            pref = 'coreg'
        else:
            pref = ''

        az_start = []
        ra_start = []
        az_end = []
        ra_end = []

        az_step = 1 / float(meta_slices[0].processes[pref + 'readfiles']['Pulse_Repetition_Frequency (computed, Hz)'])
        ra_step = 1 / float(meta_slices[0].processes[pref + 'readfiles']['Range_sampling_rate (computed, MHz)']) / 1000000

        for slice in meta_slices:
            az_datetime = slice.processes[pref + 'readfiles']['First_pixel_azimuth_time (UTC)']
            az_time = (datetime.datetime.strptime(az_datetime, '%Y-%m-%dT%H:%M:%S.%f') -
                       datetime.datetime.strptime(az_datetime[:10], '%Y-%m-%d'))

            az0 = az_time.seconds + az_time.microseconds / 1000000.0
            ra0 = float(slice.processes[pref + 'readfiles']['Range_time_to_first_pixel (2way) (ms)']) / 1000

            lines = slice.processes[pref + 'crop']['Data_first_line'] + slice.processes[pref + 'crop']['Data_lines']
            pixels = slice.processes[pref + 'crop']['Data_first_pixel'] + slice.processes[pref + 'crop']['Data_pixels']

            az_start.append(az0)
            ra_start.append(ra0)
            az_end.append(az0 + lines * az_step)
            ra_end.append(ra0 + pixels * ra_step)

        min_az = np.min(np.array(az_start))
        min_ra = np.min(np.array(ra_start))
        max_az = np.max(np.array(az_end))
        max_ra = np.max(np.array(ra_end))

        shape = [int((max_az - min_az) / az_step), int((max_ra - min_ra) / ra_step)]

        return min_az, min_ra, shape

    @staticmethod
    def find_slice_coordinates(meta_slices, coordinates, slice_offset=''):
        # This function is used to create an oversight of the coordinates of the available slices. This results in an
        # oversight of the start pixel/line of every burst compared to the full image.
        # This function can be run using the base readfiles and crop data of the coregistration image.
        # (slave images will resample to this coregistration image)

        for slice in meta_slices:
            if not isinstance(slice, ImageData):
                print('Slices should be an ImageData instance')
                return
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem instance')
            return
        if len(slice_offset) == 0:
            slice_offset = [10, 100]

        # Create the concat meta file
        meta = Concatenate.create_concat_meta(meta_slices)

        # Load the data for the provided coordinate system
        if coordinates.grid_type == 'geographic':
            if coordinates.shape == '' or coordinates.lat0 == '' or coordinates.lon0 == '':
                coordinates.add_res_info(meta)
        if coordinates.grid_type == 'projection':
            if coordinates.shape == '' or coordinates.x0 == '' or coordinates.y0 == '':
                coordinates.add_res_info(meta)
        elif coordinates.grid_type == 'radar_coordinates':
            coordinates.add_res_info(meta)

        # Create output coordinate systems
        coordinates_slices = [copy.deepcopy(coordinates) for slice in meta_slices]

        # First pixels of slices (only relevant for radar coordinates later on.)
        slices_start = []

        # Get the coordinates of the first line/pixel within the full image.
        for slice, coordinates_slice in meta_slices, coordinates_slices:
            coordinates_slice.add_res_info(slice, change_ref=False)
            slices_start.append([coordinates_slice.first_line, coordinates_slice.first_pixel])

        # For the radar coordinates calculate the new offsets
        if coordinates.grid_type == 'radar_coordinates':
            slices_offsets = FindCoordinates.find_slices_offset(coordinates.shape, coordinates.multilook,
                                                                coordinates.oversample,  coordinates.offset,
                                                                slices_start, slice_offset=slice_offset)

            for slice_offset, coordinates_slice in zip(slices_offsets, coordinates_slices):
                coordinates_slice.offset = slice_offset

        return meta, coordinates_slices

    @staticmethod
    def load_slices(step, slices, slices_coordinates, file_type=''):
        # This function checks whether the needed multilooked data is available and throws an error if not.
        # Files are loaded from memory if possible otherwise they will be loaded from disk.

        if not file_type:
            file_type = step

        # Init list of concatenate images.
        image_dat = []
        d_type = ''
        d_type_str = ''

        for slice, coordinates in zip(slices, slices_coordinates):

            type_dat = file_type + coordinates.sample
            d_type_str = slice.data_types[step][type_dat]
            d_type = slice.dtype_numpy[slice.data_types[step][type_dat]]

            if step in slice.data_memory.keys():
                if type_dat in slice.data_memory[step].keys():
                    if slice.check_coverage(step, 0, 0, coordinates.shape, loc='memory', file_type=type_dat):
                        image_dat.append(slice.data_memory[step][type_dat])
                        continue

            if step in slice.data_disk.keys():
                if type_dat in slice.data_disk[step].keys():
                    if slice.check_loaded(step, loc='disk', file_type=type_dat, warn=False):
                        image_dat.append(slice.data_disk[step][type_dat])
                        continue

                    elif slice.read_data_memmap(step, file_type):
                        continue

            print('Datafile for slice ' + os.path.basename(slice.folder) + ' in step ' + step + ' with filename ' +
                  type_dat + ' does not exist for concatenation')


        return image_dat, d_type, d_type_str