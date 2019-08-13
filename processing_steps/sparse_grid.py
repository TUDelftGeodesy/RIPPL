# The following class creates an interferogram from a master and slave image.

from rippl.meta_data.image_data import ImageData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from collections import OrderedDict, defaultdict
import logging
import os


class SparseGrid(object):

    def __init__(self, meta, step, file_type, coordinates, coor_in):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.
        # When you want to use a certain projection, please give the proj4 string to do the conversion. Most projection
        # descriptions can be found at: spatialreference.org
        # The projection name is used as a shortname for the .res file, to get track of the output files.

        if isinstance(meta, ImageData):
            self.meta = meta
        else:
            return

        self.coor_out = coordinates
        self.coor_in = coor_in

        # Load input data.
        self.step = step
        if len(file_type) == 0:
            self.file_type = step
        else:
            self.file_type = file_type
        self.meta.read_data_memmap(self.step, self.file_type)
        self.data = self.meta.data_disk[self.step][self.file_type + self.coor_in.sample]

        # Load the sparsing mask
        self.point_shape = self.coor_out.shape
        self.meta.read_data_memmap('sparse_grid', 'line' + self.coor_out.sample)
        self.meta.read_data_memmap('sparse_grid', 'pixel' + self.coor_out.sample)
        self.lines = self.meta.data_disk['sparse_grid']['line' + self.coor_out.sample]
        self.pixels = self.meta.data_disk['sparse_grid']['pixel' + self.coor_out.sample]

        # Prepare output
        self.sparse = []

    def __call__(self):
        if len(self.data) == 0 or len(self.pixels) == 0 or len(self.lines) == 0:
            print('Missing input data for sparsing for ' + self.meta.folder + '. Aborting..')
            return False

        try:
            # Save meta data and results
            self.add_meta_data(self.meta, self.coor_out, self.coor_in, self.step, self.file_type)
            self.meta.images_create_disk(self.step, self.file_type, self.coor_out)
            self.meta.read_data_memmap(self.step, self.file_type + self.coor_out.sample)
            self.sparse = self.meta.data_disk[self.step][self.file_type + self.coor_out.sample]
            self.sparse[:, :] = self.data[self.lines - self.coor_out.first_line, self.pixels - self.coor_out.first_pixel]

            return True

        except Exception:
            log_file = os.path.join(self.meta.folder, 'error.log')
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed multilooking for ' + self.meta.folder + '. Check ' + log_file + ' for details.')
            print('Failed multilooking for ' + self.meta.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def processing_info(coor_out, coor_in='', step='earth_topo_phase', file_type=''):
        # Information on this processing step. meta type should be defined here because this method is not directly
        # connected to either slave/master/ifg/coreg_master data type.

        if not isinstance(coor_out, CoordinateSystem) or not isinstance(coor_in, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        if file_type == '':
            file_type = step

        # Three input files needed x, y, z coordinates
        recursive_dict = lambda: defaultdict(recursive_dict)
        input_dat = recursive_dict()

        input_dat['slave'][step][file_type + coor_in.sample]['file'] = file_type + coor_in.sample + '.raw'
        input_dat['slave'][step][file_type + coor_in.sample]['coordinates'] = coor_in
        input_dat['slave'][step][file_type + coor_in.sample]['slice'] = coor_in.slice

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
