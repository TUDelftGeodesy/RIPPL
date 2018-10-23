# The following class creates an interferogram from a master and slave image.

from image_data import ImageData
from multilook import Multilook
from coordinate_system import CoordinateSystem
from collections import OrderedDict, defaultdict
import os
import numpy as np
import copy
import logging


class Interfero(object):

    """
    :type s_pix = int
    :type s_lin = int
    """

    def __init__(self, meta, master_meta, coordinates, coor_out='', cmaster_meta='', ifg_meta='', s_lin=0, s_pix=0, lines=0,
                 step='earth_topo_phase', file_type=''):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        if isinstance(meta, ImageData) and isinstance(master_meta, ImageData):
            self.slave = meta
            self.master = master_meta
        else:
            return

        if isinstance(cmaster_meta, ImageData):
            self.cmaster = cmaster_meta

        if isinstance(ifg_meta, ImageData):
            self.ifg = ifg_meta
        else:
            self.add_meta_data(master_meta, meta)

        if isinstance(coordinates, CoordinateSystem):
            self.coor_in = coordinates

        if isinstance(coor_out, CoordinateSystem):
            self.coor_out = coor_out
        else:
            self.coor_out = self.coor_in

        # Load data (somewhat complicated for the geographical multilooking.
        if self.coor_in.grid_type == 'radar_coordinates' and self.coor_out.grid_type in ['geographic', 'projection']:

            self.use_ids = True

            # Check additional information on geographical multilooking
            convert_sample = coor_in.sample + '_' + coor_out.sample

            sort_ids_shape = self.cmaster.image_get_data_size(step, 'sort_ids' + convert_sample)
            sum_ids_shape = self.cmaster.image_get_data_size(step, 'sum_ids' + convert_sample)
            output_ids_shape = self.cmaster.image_get_data_size(step, 'output_ids' + convert_sample)

            self.sort_ids = self.cmaster.image_load_data_memory('geocode', 0, 0, sort_ids_shape,
                                                                file_type='sort_ids' + convert_sample)
            self.sum_ids = self.cmaster.image_load_data_memory('geocode', 0, 0, sum_ids_shape,
                                                               file_type='sum_ids' + convert_sample)
            self.output_ids = self.cmaster.image_load_data_memory('geocode', 0, 0, output_ids_shape,
                                                                  file_type='output_ids' + convert_sample)

        elif self.coor_out.grid_type != self.coor_in.grid_type:
            print('Conversion from geographic or projection grid type to other grid type not supported. Aborting')
            return
        else:
            self.use_ids = False

        # Input data
        self.step = step
        if file_type == '':
            self.file_type = step
        else:
            self.file_type = file_type

        # Currently not possible to perform this step in slices because it includes multilooking. Maybe this will be
        # able later on. (Convert to different grids and slicing can cause problems at the sides of the slices.)
        self.master_dat = self.master.image_load_data_memory(self.step, 0, 0, coor_in.shape, self.step, warn=False)
        self.slave_dat = self.slave.image_load_data_memory('earth_topo_phase', 0, 0, coor_in.shape, 'earth_topo_phase', warn=False)

        self.interferogram = []

    def __call__(self):
        # Check if needed data is loaded
        if len(self.slave_dat) == 0 or len(self.master_dat) == 0:
            print('Missing input data for creating interferogram for ' + self.ifg.folder + '. Aborting..')
            return False

        # This function is a special case as it allows the creation of the interferogram together with multilooking
        # of this interferogram in one step. It is therefore a kind of nested function.
        # This is done like this because it prevents the creation of intermediate interferograms which can take a large
        # part of memory or disk space.
        # For the coherence this is not needed, therefore a different approach is used there.

        try:

            self.interferogram = self.master_dat * self.slave_dat.conj()

            # Calculate the multilooked image.
            if self.coor_in.grid_type == 'radar_coordinates' and self.coor_out.grid_type in ['geographic','projection']:
                if self.coor_out.grid_type == 'geographic':
                    self.multilooked = Multilook.radar2geographical(self.interferogram, self.coor_out, self.sort_ids,
                                                                    self.sum_ids, self.output_ids)
                elif self.coor_out.grid_type == 'projection':
                    self.multilooked = Multilook.radar2projection(self.interferogram, self.coor_out, self.sort_ids,
                                                                    self.sum_ids, self.output_ids)
            # If we use the same coordinate system for input and output.
            elif self.coor_in.grid_type == self.coor_out.grid_type:

                if self.coor_in.grid_type == 'radar_coordinates':
                    self.multilooked = Multilook.radar2radar(self.interferogram, self.coor_in, self.coor_out)
                elif self.coor_in.grid_type == 'projection':
                    self.multilooked = Multilook.projection2projection(self.interferogram, self.coor_in, self.coor_out)
                elif self.coor_in.grid_type == 'geographic':
                    self.multilooked = Multilook.geographic2geographic(self.interferogram, self.coor_in, self.coor_out)
            else:
                print('Conversion from a projection or geographic coordinate system to another system is not possible')

            # Save meta data and results
            self.add_meta_data(self.ifg, self.coor_out, self.step, self.file_type)
            self.ifg.image_new_data_memory(self.multilooked, 'interferogram', 0, 0, file_type='interferogram' + self.coor_out.sample)

            return True

        except Exception:
            log_file = os.path.join(self.ifg.folder, 'error.log')
            if not os.path.exists(self.ifg.folder):
                os.makedirs(self.ifg.folder)
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception('Failed creating interferogram for ' +
                              self.ifg.folder + '. Check ' + log_file + ' for details.')
            print('Failed creating interferogram for ' +
                  self.ifg.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def create_meta_data(meta, master_meta):
        # This function creates a folder and .res file for this interferogram.
        master_path = master_meta.res_path
        slave_path = meta.res_path

        if os.path.basename(os.path.dirname(master_path)) == 8:
            master_date = os.path.basename(os.path.dirname(master_path))
            slave_date = os.path.basename(os.path.dirname(slave_path))
            ifgs_folder = os.path.join(os.path.dirname(os.path.dirname(master_path)),
                                       master_date + '_' + slave_date)
        else:   # If we are working with bursts we have to go back to the second level.
            master_date = os.path.basename(os.path.dirname(os.path.dirname(master_path)))
            slave_date = os.path.basename(os.path.dirname(os.path.dirname(slave_path)))
            ifgs_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(master_path))),
                                       master_date + '_' + slave_date,
                                       os.path.basename(os.path.dirname(master_path)))

        ifg = ImageData(filename='', res_type='interferogram')
        ifg.res_path = os.path.join(ifgs_folder, 'info.res')
        ifg.folder = ifgs_folder
        
        if 'coreg_readfiles' in master.processes.keys():
            ifg.image_add_processing_step('coreg_readfiles', copy.deepcopy(master.processes['coreg_readfiles']))
            ifg.image_add_processing_step('coreg_orbits', copy.deepcopy(master.processes['coreg_orbits']))
            ifg.image_add_processing_step('coreg_crop', copy.deepcopy(master.processes['coreg_crop']))
        elif 'coreg_readfiles' in slave.processes.keys():
            ifg.image_add_processing_step('coreg_readfiles', copy.deepcopy(slave.processes['coreg_readfiles']))
            ifg.image_add_processing_step('coreg_orbits', copy.deepcopy(slave.processes['coreg_orbits']))
            ifg.image_add_processing_step('coreg_crop', copy.deepcopy(slave.processes['coreg_crop']))

        # Derive geometry
        ifg.geometry()

        return ifg

    @staticmethod
    def add_meta_data(ifg_meta, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'interferogram' in ifg_meta.processes.keys():
            meta_info = ifg_meta.processes['interferogram']
        else:
            meta_info = OrderedDict()

        for coor in coordinates:
            if not isinstance(coor, CoordinateSystem):
                print('coordinates should be an CoordinateSystem object')
                return

            meta_info = coor.create_meta_data(['interferogram' + coor.sample], ['complex_float'])

        ifg_meta.image_add_processing_step('interferogram', meta_info)

    @staticmethod
    def processing_info(coor_out, coor_in='', ifg_input_step='earth_topo_phase', ifg_input_type='earth_topo_phase'):

        if not isinstance(coor_in, CoordinateSystem):
            coor_in = CoordinateSystem()
            coor_in.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
        if not isinstance(coor_out, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Three input files needed x, y, z coordinates
        input_dat = defaultdict()
        input_dat['slave'][ifg_input_step][ifg_input_type]['file'] = [ifg_input_type + '.raw']
        input_dat['slave'][ifg_input_step][ifg_input_type]['coordinates'] = coor_in
        input_dat['slave'][ifg_input_step][ifg_input_type]['slice'] = 'True'
        input_dat['slave'][ifg_input_step][ifg_input_type]['coor_change'] = 'multilook'

        input_dat['master'][ifg_input_step][ifg_input_type]['file'] = [ifg_input_type + '.raw']
        input_dat['master'][ifg_input_step][ifg_input_type]['coordinates'] = coor_in
        input_dat['master'][ifg_input_step][ifg_input_type]['slice'] = 'True'
        input_dat['master'][ifg_input_step][ifg_input_type]['coor_change'] = 'multilook'

        # line and pixel output files.
        output_dat = defaultdict()
        output_dat['slave']['interferogram']['interferogram']['file'] = ['interferogram' + coor_out.sample + '.raw']
        output_dat['slave']['interferogram']['interferogram']['coordinates'] = coor_out
        output_dat['slave']['interferogram']['interferogram']['slice'] = coor_out.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 2

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, file_type='', coordinates=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.
        meta.images_create_disk('interferogram', file_type, coordinates)

    @staticmethod
    def save_to_disk(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_create_disk('interferogram', file_type, coordinates)

    @staticmethod
    def clear_memory(meta, file_type='', coordinates=''):
        # Save the function output in memory to disk
        meta.images_clean_memory('interferogram', file_type, coordinates)

