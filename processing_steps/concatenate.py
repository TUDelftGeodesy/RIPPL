# This class uses a shapefile and a radar image to detect which burst are part of the image and which bursts are not
# The function has two different options:
# 1. Combine an input image and a shapefile to find the corresponding bursts.
# 2. Combine information on the specified bursts and a new image to check whether all bursts are available in this
#       image and how they are linked.


# The following class creates an interferogram from a master and slave image.

from rippl.image_data import ImageData
from rippl.find_coordinates import FindCoordinates
from rippl.coordinate_system import CoordinateSystem
from rippl.orbit_resample_functions.orbit_coordinates import OrbitCoordinates
from collections import OrderedDict, defaultdict
import datetime
import numpy as np
import os
import copy
import logging


class Concatenate(object):

    def __init__(self, meta_slices, coordinates='', meta='', step='interferogram', file_type='', out_data='disk'):
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

        if not coordinates:
            self.coordinates = CoordinateSystem()
            self.coordinates.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
        else:
            self.coordinates = coordinates

        self.out_data = out_data

        if meta == '':
            self.meta = Concatenate.create_concat_meta(meta_slices)
        elif isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        dummy, self.in_coordinates_slices, self.coordinates_slices = \
            Concatenate.find_slice_coordinates(self.meta_slices, self.coordinates, self.coordinates, step=self.step, file_type=self.file_type)

    def __call__(self):

        if len(self.meta_slices) == 0:
            print('Missing input data for concatenation of image ' + self.meta.folder + '. Aborting..')
            return False

        try:

            self.add_meta_data(self.meta_slices, self.coordinates_slices, self.meta, self.coordinates, self.step, self.file_type)

            # Load the data from the slices
            data_slices, data_type, str_data_type = Concatenate.load_slices(self.step, self.meta_slices, self.coordinates_slices, self.file_type)

            if self.out_data == 'disk':
                self.meta.image_create_disk(self.step, self.file_type + self.coordinates.sample)
                data = self.meta.data_disk[self.step][self.file_type + self.coordinates.sample]
            elif self.out_data == 'memory':
                empty_image = np.zeros(self.coordinates.shape).astype(self.meta.dtype_numpy[data_type])
                self.meta.image_new_data_memory(empty_image, self.step, 0, 0, self.file_type + self.coordinates.sample)
                data = self.meta.data_memory[self.step][self.file_type + self.coordinates.sample]
            else:
                print('out_data should either be disk or memory')
                return

            for data_slice, coordinates_slice, meta in zip(data_slices, self.coordinates_slices, self.meta_slices):

                print('Adding file type ' + self.file_type + coordinates_slice.sample + ' for step ' + self.step + ' of slice ' + os.path.basename(os.path.dirname(meta.res_path)))

                s_pix = coordinates_slice.first_pixel - 1
                s_lin = coordinates_slice.first_line - 1
                e_pix = coordinates_slice.first_pixel + coordinates_slice.shape[1] - 1
                e_lin = coordinates_slice.first_line + coordinates_slice.shape[0] - 1

                # Use different methods with different type of data. Complex data is added (radar data), while all other
                # data is simply replaced (information on geocoding or otherwise)
                if str_data_type in ['complex_int', 'complex_short', 'complex_real4', 'tiff'] or \
                        self.step in ['square_amplitude']:
                    if self.out_data == 'disk':
                        cpx_int = self.meta.dtype_disk['complex_int']
                        cpx_flt = self.meta.dtype_disk['complex_short']

                        if str_data_type == 'complex_int':
                            data[s_lin:e_lin, s_pix:e_pix] = \
                                (data.view(np.int16).astype('float32', subok=False).view(np.complex64)[s_lin:e_lin, s_pix:e_pix]
                                 + data_slice.view(np.int16).astype('float32', subok=False).view(np.complex64)
                                 ).view(np.float32).astype(np.int16).view(cpx_int)
                        elif str_data_type == 'complex_short':
                            data[s_lin:e_lin, s_pix:e_pix] = \
                                (data.view(np.float16).astype('float32', subok=False).view(np.complex64)[s_lin:e_lin, s_pix:e_pix]
                                 + data_slice.view(np.float16).astype('float32', subok=False).view(np.complex64)
                                 ).view(np.float32).astype(np.int16).view(cpx_flt)
                        else:
                            data[s_lin:e_lin, s_pix:e_pix] += data_slice

                    elif self.out_data == 'memory':
                        data[s_lin:e_lin, s_pix:e_pix] += data_slice
                else:
                    data[s_lin:e_lin, s_pix:e_pix] = data_slice

            for slice in self.meta_slices:
                slice.clean_memmap_files()
                slice.clean_memory()

            if self.out_data == 'disk':
                data.flush()

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed concatenation for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed concatenation for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def add_meta_data(meta_slices, meta_coordinates, meta, coordinates, step, file_type=''):
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

        data_type = meta_slices[0].data_types[step][file_type + meta_coordinates[0].sample]
        meta_info = coordinates.create_meta_data([file_type], [data_type], meta_info)
        meta.image_add_processing_step(step, meta_info)

    @staticmethod
    def processing_info(meta_slices, coordinates, step, meta_type='', file_type=''):
        # This is a special case where we go from slices to full images. Therefore we need some extra info to find
        # the input/output information. This function needs the input slices including the information of the radar
        # coordinates of the original or coreg grid (depending on what is needed.)

        # For multiprocessing it is important that coreg master grid is used.
        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        meta, coordinates_slices, in_coordinates_slices = Concatenate.find_slice_coordinates(meta_slices, coordinates)
        slice_names = [os.path.basename(slice.folder) for slice in meta_slices]
        slice_file_names = [file_type + coor.sample + '.raw' for coor in coordinates_slices]

        # Three input files needed x, y, z coordinates
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()
        input_dat['meta'][step][file_type]['files'] = slice_file_names
        input_dat['meta'][step][file_type]['slice_names'] = slice_names
        input_dat['meta'][step][file_type]['coordinates'] = coordinates_slices
        input_dat['meta'][step][file_type]['slice'] = True

        # line and pixel output files.
        output_dat = recursive_dict()
        output_dat['meta'][step][file_type + coordinates.sample]['file'] = file_type + coordinates.sample + '.raw'
        output_dat['meta'][step][file_type + coordinates.sample]['coordinates'] = coordinates
        output_dat['meta'][step][file_type + coordinates.sample]['slice'] = 'False'

        # Data is used only once for concatenation.
        mem_use = 1

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, step, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk(step, file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, step, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_memory_to_disk(step, file_type, coordinates)

    @staticmethod
    def clear_memory(meta, step, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory(step, file_type, coordinates)

    @staticmethod
    def get_concat_bounding_box(min_az, min_ra, shape, slice):
        # To get a good approximation of the final size of our image we rotate the coordinates to an approximate
        # azimuth/range system.

        orbit = OrbitCoordinates(slice)
        orbit.ra_time = min_ra
        orbit.az_seconds = min_az

        orbit.lp_time(lines=[0, 0, shape[0], shape[0]], pixels=[0, shape[1], shape[1], 0], regular=False)
        orbit.height = np.array([0, 0, 0, 0])

        orbit.lph2xyz()
        orbit.xyz2ell()

        return orbit.lon, orbit.lat

    @staticmethod
    def create_concat_meta(meta_slices):
        # This code create a new concatenated slice.

        # First define the new folder and .res name
        filename = os.path.join(os.path.dirname(os.path.dirname(meta_slices[0].res_path)), 'info.res')

        # Then create a single or interferogram file
        meta_type = meta_slices[0].res_type
        meta = ImageData(filename, meta_type)

        min_az, min_ra, shape = Concatenate.find_radar_max_extend(meta_slices)

        lon_corners, lat_corners = Concatenate.get_concat_bounding_box(min_az, min_ra, shape, meta_slices[0])

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
            meta.processes[step_meta + 'crop']['crop_output_file'] = 'crop.raw'
            meta.processes[step_meta + 'crop']['crop_output_format'] = 'complex_int'
            meta.processes[step_meta + 'crop']['crop_first_line'] = '1'
            meta.processes[step_meta + 'crop']['crop_first_pixel'] = '1'
            meta.processes[step_meta + 'crop']['crop_lines'] = str(int(shape[0]))
            meta.processes[step_meta + 'crop']['crop_pixels'] = str(int(shape[1]))
            meta.processes[step_meta + 'crop'].pop('crop_first_line (w.r.t. tiff_image)')
            meta.processes[step_meta + 'crop'].pop('crop_last_line (w.r.t. tiff_image)')

            # Change from slice to full image
            meta.processes[step_meta + 'readfiles']['slice'] = 'False'

            # Add coordinates and azimuth/range timing
            meta.processes[step_meta + 'readfiles']['Number_of_lines_original'] = str(int(shape[0]))
            meta.processes[step_meta + 'readfiles']['Number_of_pixels_original'] = str(int(shape[1]))
            meta.processes[step_meta + 'readfiles']['First_pixel_azimuth_time (UTC)'] = az_time
            meta.processes[step_meta + 'readfiles']['Range_time_to_first_pixel (2way) (ms)'] = ra_time

            # Remove all burst specific information to avoid confusion
            meta.processes[step_meta + 'readfiles'].pop('Number_of_lines_Swath')
            meta.processes[step_meta + 'readfiles'].pop('SWATH')
            meta.processes[step_meta + 'readfiles'].pop('number_of_pixels_Swath')
            meta.processes[step_meta + 'readfiles'].pop('total_Burst')
            meta.processes[step_meta + 'readfiles'].pop('Burst_number_index')

            # Add the new coordinates. This is only an approximation using a combination of the burst lat/lon coordinates
            meta.processes[step_meta + 'readfiles']['Scene_ul_corner_latitude'] = str(lat_corners[0])
            meta.processes[step_meta + 'readfiles']['Scene_ur_corner_latitude'] = str(lat_corners[1])
            meta.processes[step_meta + 'readfiles']['Scene_lr_corner_latitude'] = str(lat_corners[2])
            meta.processes[step_meta + 'readfiles']['Scene_ll_corner_latitude'] = str(lat_corners[3])
            meta.processes[step_meta + 'readfiles']['Scene_ul_corner_longitude'] = str(lon_corners[0])
            meta.processes[step_meta + 'readfiles']['Scene_ur_corner_longitude'] = str(lon_corners[1])
            meta.processes[step_meta + 'readfiles']['Scene_lr_corner_longitude'] = str(lon_corners[2])
            meta.processes[step_meta + 'readfiles']['Scene_ll_corner_longitude'] = str(lon_corners[3])

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

        meta.geometry()

        return meta

    @staticmethod
    def find_radar_max_extend(meta_slices):
        # Finds the maximum extend of the image in original line, pixel coordinates.

        if meta_slices[0].process_control['coreg_readfiles'] == '1':
            pref = 'coreg_'
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

            lines = int(slice.processes[pref + 'crop']['crop_first_line']) + int(slice.processes[pref + 'crop']['crop_lines'])
            pixels = int(slice.processes[pref + 'crop']['crop_first_pixel']) + int(slice.processes[pref + 'crop']['crop_pixels'])

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
    def find_slice_coordinates(meta_slices, in_coor, out_coor, slice_offset='', step='', file_type=''):
        # This function is used to create an oversight of the coordinates of the available slices. This results in an
        # oversight of the start pixel/line of every burst compared to the full image.
        # This function can be run using the base readfiles and crop data of the coregistration image.
        # (slave images will resample to this coregistration image)

        for slice in meta_slices:
            if not isinstance(slice, ImageData):
                print('Slices should be an ImageData instance')
                return
        if not isinstance(in_coor, CoordinateSystem) and not isinstance(out_coor, CoordinateSystem):
            print('coordinates should be an CoordinateSystem instance')
            return
        if len(slice_offset) == 0:
            slice_offset = [0, 0]

        # Create the concat meta file
        meta = Concatenate.create_concat_meta(meta_slices)

        # Load the data for the provided coordinate system
        if out_coor.grid_type == 'geographic':
            if len(out_coor.shape) == 0 or out_coor.lat0 == '' or out_coor.lon0 == '':
                out_coor.add_res_info(meta)
        if out_coor.grid_type == 'projection':
            if len(out_coor.shape) == 0 or out_coor.x0 == '' or out_coor.y0 == '':
                out_coor.add_res_info(meta)
        elif out_coor.grid_type == 'radar_coordinates':
            out_coor.add_res_info(meta)

        if in_coor.grid_type == 'geographic':
            if len(in_coor.shape) == 0 or in_coor.lat0 == '' or in_coor.lon0 == '':
                in_coor.add_res_info(meta)
        if in_coor.grid_type == 'projection':
            if len(in_coor.shape) == 0 or in_coor.x0 == '' or in_coor.y0 == '':
                in_coor.add_res_info(meta)
        elif in_coor.grid_type == 'radar_coordinates':
            in_coor.add_res_info(meta)

        # Create output coordinate systems
        in_coor_slices = [copy.deepcopy(in_coor) for slice in meta_slices]
        out_coor_slices = [copy.deepcopy(out_coor) for slice in meta_slices]

        # First pixels of slices (only relevant for radar coordinates later on.)
        slices_start = []

        # Get the coordinates of the first line/pixel within the full image.
        coordinates_slices = []
        out_coordinates_slices = []
        in_coordinates_slices = []

        for slice, in_coor_slice, out_coor_slice in zip(meta_slices, in_coor_slices, out_coor_slices):
            if step == '' or file_type == '':
                in_coor_slice.add_res_info(slice, change_ref=False)
                if out_coor_slice.grid_type == 'radar_coordinates':
                    out_coor_slice.add_res_info(slice, change_ref=False)
            else:
                old_coor = slice.read_res_coordinates(step)[-1]
                in_coor_slice.add_res_info(slice, change_ref=False, old_coor=old_coor)
                if out_coor_slice.grid_type == 'radar_coordinates':
                    out_coor_slice.add_res_info(slice, change_ref=False, old_coor=old_coor)

            in_coordinates_slices.append(in_coor_slice)
            coordinates_slices.append(out_coor_slice)
            slices_start.append([in_coor_slice.first_line, in_coor_slice.first_pixel])

        # For the radar coordinates calculate the new offsets
        if out_coor.grid_type == 'radar_coordinates':
            slices_offsets = FindCoordinates.find_slices_offset(out_coor.shape, 0, 0, out_coor.multilook,
                                                                out_coor.oversample,  out_coor.offset,
                                                                slices_start, slice_offset=slice_offset)

            for slice, slice_offset, slice_start, coordinates_slice in zip(meta_slices, slices_offsets, slices_start, coordinates_slices):
                coordinates_slice.offset = slice_offset

                if step == '' or file_type == '':
                    coordinates_slice.add_res_info(slice, change_ref=False)
                else:
                    old_coor = slice.read_res_coordinates(step)[-1]

                    coordinates_slice.add_res_info(slice, change_ref=False, old_coor=old_coor)
                coordinates_slice.first_pixel = (slice_start[1] + slice_offset[1]) // out_coor.multilook[1] + 1
                coordinates_slice.first_line = (slice_start[0] + slice_offset[0]) // out_coor.multilook[0] + 1
                sample = FindCoordinates.multilook_str(coordinates_slice.multilook, coordinates_slice.oversample, coordinates_slice.offset)[0]
                coordinates_slice.sample = sample
                out_coordinates_slices.append(coordinates_slice)
        else:
            for slice, slice_start, coordinates_slice in zip(meta_slices, slices_start, coordinates_slices):
                coordinates_slice.offset = slice_offset

                """
                if step == '' or file_type == '':
                    coordinates_slice.add_res_info(slice, change_ref=False)
                else:
                    old_coor = slice.read_res_coordinates(step)[-1]
                    coordinates_slice.add_res_info(slice, change_ref=False, old_coor=old_coor)

                coordinates_slice.first_pixel = int(slice_start[1]) + 1
                coordinates_slice.first_line = int(slice_start[0]) + 1
                coordinates_slice.sample = out_coor.sample
                """

                out_coordinates_slices.append(coordinates_slice)

        return meta, in_coordinates_slices, out_coordinates_slices

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

                    elif slice.read_data_memmap(step, type_dat):
                        image_dat.append(slice.data_disk[step][type_dat])
                        continue

            print('Datafile for slice ' + os.path.basename(slice.folder) + ' in step ' + step + ' with filename ' +
                  type_dat + ' does not exist for concatenation')


        return image_dat, d_type, d_type_str