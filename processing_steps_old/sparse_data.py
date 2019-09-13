# The following class creates an interferogram from a master and slave image.

from rippl.meta_data.image_processing_data import ImageData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from collections import OrderedDict, defaultdict
import logging
import os


class SparseData(object):

    def __init__(self, meta, cmaster_meta, step, file_type, coordinates, coor_in=''):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.
        # When you want to use a certain projection, please give the proj4 string to do the conversion. Most projection
        # descriptions can be found at: spatialreference.org
        # The projection name is used as a shortname for the .res file, to get track of the output files.

        if isinstance(meta, ImageData) and isinstance(cmaster_meta, ImageData):
            self.meta = meta
            self.cmaster = cmaster_meta
        else:
            return

        # If the in coordinates are not defined, we default to the original radar coordinate system. This is the obvious
        # choiche
        if not isinstance(coor_in, CoordinateSystem):
            self.coor_in = CoordinateSystem()
            self.coor_in.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
        else:
            self.coor_in = coor_in
        if isinstance(coordinates, CoordinateSystem):
            self.coor_out = coordinates

        # Check if input and output coordinate systems match
        if not self.coor_out.sparse_grid:
            print('The output coordinate system for sparse data should be a sparse grid')
            return
        elif self.coor_out.sample != self.coor_in.sample + '_' + self.coor_out.sparse_name:
            print('The output and input coordinate systems do not match')
            return

        # Load input data.
        self.meta.read_data_memmap(step, file_type + self.coor_in.sample)
        self.data = self.meta.data_disk[step][file_type + self.coor_in.sample]

        # Load lines and pixels
        pix_shape = self.cmaster.data_sizes['point_data']['line' + self.coor_out.sample]
        self.coor_out.shape = pix_shape

        self.line = self.cmaster.image_load_data_memory('point_data', 0, 0, pix_shape, 'line' + self.coor_out.sample)
        self.pixel = self.cmaster.image_load_data_memory('point_data', 0, 0, pix_shape, 'pixel' + self.coor_out.sample)

        # Get offset and multilook
        offset = [int(self.meta.processes[step][file_type + self.coor_in.sample + '_first_pixel']) - 1 +
                  int(self.meta.processes[step][file_type + self.coor_in.sample + '_offset_range']),
                  int(self.meta.processes[step][file_type + self.coor_in.sample + '_first_line']) - 1 +
                  int(self.meta.processes[step][file_type + self.coor_in.sample + '_offset_azimuth'])]
        multilook = [int(self.meta.processes[step][file_type + self.coor_in.sample + '_multilook_range']),
                     int(self.meta.processes[step][file_type + self.coor_in.sample + '_multilook_azimuth'])]

        self.pixel = (self.pixel - offset[0]) // multilook[0]
        self.line = (self.line - offset[1]) // multilook[1]

        self.step = step
        if len(file_type) == 0:
            self.file_type = step
        else:
            self.file_type = file_type

        # Prepare output
        self.sparse = []

    def __call__(self):
        if len(self.data) == 0 or len(self.line) == 0 or len(self.pixel) == 0:
            print('Missing input data for multilooking for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Calculate the multilooked image.
            self.sparse = self.data[self.line, self.pixel]

            # Save meta data and results
            self.add_meta_data(self.meta, self.coor_out, self.coor_in, self.step, self.file_type)
            self.meta.image_new_data_memory(self.sparse, self.step, 0, 0, file_type=self.file_type + self.coor_out.sample)

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed sparsing for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed sparsing for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def processing_info(coor_out, coor_in='', step='earth_topo_phase', file_type=''):
        # Information on this processing step. meta type should be defined here because this method is not directly
        # connected to either slave/master/ifg/coreg_master data type.

        if not isinstance(coor_in, CoordinateSystem):
            coor_in = CoordinateSystem()
            coor_in.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
        if not isinstance(coor_out, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if file_type == '':
            file_type = step

        # Three input files needed x, y, z coordinates
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()

        input_dat['slave'][step][file_type + coor_in.sample]['file'] = file_type + coor_in.sample + '.raw'
        input_dat['slave'][step][file_type + coor_in.sample]['coordinates'] = coor_in
        input_dat['slave'][step][file_type + coor_in.sample]['slice'] = coor_in.slice

        if coor_out.sparse_grid:
            for dat_type in ['line', 'pixel']:
                input_dat['cmaster']['point_data'][dat_type + coor_out.sample]['files'] = dat_type + coor_out.sample + '.raw'
                input_dat['cmaster']['point_data'][dat_type + coor_out.sample]['coordinates'] = coor_out
                input_dat['cmaster']['point_data'][dat_type + coor_out.sample]['slice'] = coor_out.slice

        # line and pixel output files.
        output_dat = recursive_dict()
        input_dat['slave'][step][file_type + coor_out.sample]['file'] = file_type + coor_out.sample + '.raw'
        input_dat['slave'][step][file_type + coor_out.sample]['coordinates'] = coor_out
        input_dat['slave'][step][file_type + coor_out.sample]['slice'] = coor_out.slice

        # Number of times input data is used in ram.
        mem_use = 2

        return input_dat, output_dat, mem_use

    @staticmethod
    def add_meta_data(meta, coordinates, coor_in, step, file_type=''):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if not isinstance(coordinates, CoordinateSystem) or not isinstance(coor_in, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if step in meta.processes.keys():
            meta_info = meta.processes[step]
        else:
            meta_info = OrderedDict()

        if not file_type:
            file_type = step

        data_types = [meta.data_types[step][file_type + coor_in.sample]]
        meta_info = coordinates.create_meta_data([file_type], data_types, meta_info)
        meta.image_add_processing_step(step, meta_info)

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


