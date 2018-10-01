# The following class creates an interferogram from a master and slave image.

from image_data import ImageData
from collections import OrderedDict, defaultdict
import numpy as np
from processing_steps.interfero import Interfero
from find_coordinates import FindCoordinates
from coordinate_system import CoordinateSystem
import logging
import os


class Coherence(object):

    """
    :type s_pix = int
    :type s_lin = int
    """

    def __init__(self, master_meta, slave_meta, coordinates, ifg_meta='', s_lin=0, s_pix=0, lines=0):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        if isinstance(slave_meta, ImageData) and isinstance(master_meta, ImageData):
            self.slave = slave_meta
            self.master = master_meta
        else:
            return

        if isinstance(ifg_meta, ImageData):
            self.ifg = ifg_meta
        else:
            Interfero.create_meta_data(master_meta, slave_meta)

        if isinstance(coordinates, CoordinateSystem):
            self.coordinates = [coordinates]
        else:
            self.coordinates = coordinates

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        self.in_s_lin = []
        self.in_s_pix = []
        self.in_shapes = []
        self.out_s_lin = [s_lin for coor in coordinates]
        self.out_s_pix = [s_pix for coor in coordinates]
        self.out_shape = []

        for coor in coordinates:

            smp, ml, ovr, off, [in_s_lin, in_s_pix, in_shape], [s_lin, s_pix, shape] = \
                FindCoordinates.multilook_coors(coor.shape, s_lin=s_lin, s_pix=s_pix, lines=lines)

            self.in_s_lin.append(in_s_lin)
            self.in_s_pix.append(in_s_pix)
            self.in_shapes.append(in_shape)
            self.out_shape.append(shape)

        # Define the region that should be read in memory.
        self.s_pix_min = np.min(np.array(self.in_s_pix))
        self.s_lin_min = np.min(np.array(self.in_s_lin))
        self.in_shape = [
            np.max(np.array([i[0] for i in self.in_shapes]) + np.array(self.in_s_lin)) - self.s_lin_min,
            np.max(np.array([i[1] for i in self.in_shapes]) + np.array(self.in_s_pix)) - self.s_pix_min]

        # Check whether one image is originally referenced to the other.
        self.master_dat = self.master.image_load_data_memory('earth_topo_phase', self.s_lin_min, self.s_pix_min, self.in_shape, 'Data', warn=False)

        if len(self.master_dat) == 0:
            # In case the master date is the same as the date for the referenced image.
            ref_dat = self.slave.processes['combined_coreg']['Master reference date']

            if self.master.processes['readfiles']['First_pixel_azimuth_time (UTC)'][:10] == ref_dat:
                self.master_dat = self.master.image_load_data_memory('crop', self.s_lin_min, self.s_pix_min, self.in_shape, 'Data')

        # Check whether one image is originally referenced to the other.
        self.slave_dat = self.slave.image_load_data_memory('earth_topo_phase', self.s_lin_min, self.s_pix_min, self.in_shape, 'Data', warn=False)

        if len(self.slave_dat) == 0:
            # In case the slave date is the same as the date for the referenced image.
            ref_dat = self.master.processes['combined_coreg']['Master reference date']

            if self.slave.processes['readfiles']['First_pixel_azimuth_time (UTC)'][:10] == ref_dat:
                self.slave_dat = self.slave.image_load_data_memory('crop', self.s_lin_min, self.s_pix_min, self.in_shape, 'Data')

        self.coherence = dict()

    def __call__(self):
        # Check if needed data is loaded
        if len(self.master_dat) == 0 or len(self.slave_dat) == 0:
            print('Missing input data for processing coherence for ' + self.ifg.folder + '. Aborting..')
            return False

        try:
            # Calculate the new line and pixel coordinates based orbits / geometry
            for coor, min_pix, min_line, shape in \
                    zip(self.coordinates, self.in_s_pix, self.in_s_lin, self.in_shapes):
                if coor.multilook == [1, 1]:
                    self.coherence[coor.sample] = self.master_dat * self.slave_dat.conj()
                elif coor.oversampling == [1, 1]:
                    ir = np.arange(0, shape[0], coor.multilook[0])
                    jr = np.arange(0, shape[1], coor.multilook[1])

                    ls = min_line - self.s_lin_min
                    ps = min_pix - self.s_pix_min
                    le = ls + shape[0]
                    pe = ps + shape[1]

                    self.coherence[coor.sample] = np.abs(np.add.reduceat(np.add.reduceat(self.master_dat[ls:le, ps:pe] * self.slave_dat[ls:le, ps:pe].conj(), ir), jr, axis=1)) / \
                             np.sqrt(np.add.reduceat(np.add.reduceat(np.abs(self.master_dat[ls:le, ps:pe])**2, ir), jr, axis=1) *
                             np.add.reduceat(np.add.reduceat(np.abs(self.slave_dat[ls:le, ps:pe])**2, ir), jr, axis=1))
                else:
                    ir_h = np.arange(0, shape[0], coor.multilook[0] / coor.oversampling[0])[:-(coor.oversampling[0] - 1)]
                    ir = [[i, i + coor.multilook[0]] for i in ir_h]
                    ir = [item for sublist in ir for item in sublist][:-1]
                    jr_h = np.arange(0, shape[1], coor.multilook[1] / coor.oversampling[1])[:-(coor.oversampling[1] - 1)]
                    jr = [[i, i + coor.multilook[1]] for i in jr_h]
                    jr = [item for sublist in jr for item in sublist][:-1]

                    ls = min_line - self.s_lin_min
                    ps = min_pix - self.s_pix_min
                    le = ls + shape[0]
                    pe = ps + shape[1]

                    self.coherence[coor.sample] = np.abs(np.add.reduceat(np.add.reduceat(self.master_dat[ls:le, ps:pe] * self.slave_dat[ls:le, ps:pe].conj(), ir)[::2, :], jr, axis=1)[:, ::2]) / \
                             np.sqrt(np.add.reduceat(np.add.reduceat(np.abs(self.master_dat[ls:le, ps:pe])**2, ir)[::2, :], jr, axis=1)[:, ::2] *
                             np.add.reduceat(np.add.reduceat(np.abs(self.slave_dat[ls:le, ps:pe])**2, ir)[::2, :], jr, axis=1)[:, ::2])

            # If needed do the multilooking step
            self.add_meta_data(self.ifg, self.coordinates)

            for out_s_lin, out_s_pix, coor in zip(self.out_s_lin, self.out_s_pix, self.coordinates):
                self.ifg.image_new_data_memory(self.coherence[coor.sample], 'coherence', out_s_lin, out_s_pix, 'coherence' + coor.sample)

            return True

        except Exception:
            log_file = os.path.join(self.ifg.folder, 'error.log')
            if not os.path.exists(self.ifg.folder):
                os.makedirs(self.ifg.folder)
            logging.basicConfig(filename=log_file, level=logging.DEBUG)
            logging.exception(
                'Failed processing coherence for ' + self.ifg.folder + '. Check ' + log_file + ' for details.')
            print('Failed processing coherence for ' + self.ifg.folder + '. Check ' + log_file + ' for details.')

            return False

    @staticmethod
    def add_meta_data(ifg, coordinates):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'coherence' in ifg.processes.keys():
            meta_info = ifg.processes['coherence']
        else:
            meta_info = OrderedDict()

        for coor in coordinates:
            if not isinstance(coor, CoordinateSystem):
                print('coordinates should be an CoordinateSystem object')
                return

            meta_info = coor.create_meta_data(['coherence' + coor.sample], ['complex_float'])

        ifg.image_add_processing_step('coherence', meta_info)

    @staticmethod
    def processing_info(coordinates, ifg_input_step='earth_topo_phase', ifg_input_type='earth_topo_phase'):

        if not isinstance(coordinates, CoordinateSystem):
            print('coordinates should be an CoordinateSystem object')

        # Three input files needed x, y, z coordinates
        input_dat = defaultdict()
        coor = CoordinateSystem()
        coor.create_radar_coordinates(multilook=[1, 1], offset=[0, 0], oversample=[1, 1])
        input_dat['slave'][ifg_input_step][ifg_input_type]['file'] = [ifg_input_type + '.raw']
        input_dat['slave'][ifg_input_step][ifg_input_type]['coordinates'] = coor
        input_dat['slave'][ifg_input_step][ifg_input_type]['slice'] = coor.slice

        input_dat['master'][ifg_input_step][ifg_input_type]['file'] = [ifg_input_type + '.raw']
        input_dat['master'][ifg_input_step][ifg_input_type]['coordinates'] = coor
        input_dat['master'][ifg_input_step][ifg_input_type]['slice'] = coor.slice

        # line and pixel output files.
        output_dat = defaultdict()
        for coor in coordinates:
            output_dat['ifg']['coherence']['coherence']['file'] = ['coherence' + coor.sample + '.raw']
            output_dat['ifg']['coherence']['coherence']['coordinates'] = coor
            output_dat['ifg']['coherence']['coherence']['slice'] = coor.slice

        # Number of times input data is used in ram. Bit difficult here but 20 times is ok guess.
        mem_use = 20

        return input_dat, output_dat, mem_use

    @staticmethod
    def create_output_files(meta, output_file_steps=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not output_file_steps:
            meta_info = meta.processes['coherence']
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            output_file_steps = [filename[:-13] for filename in output_file_keys]

        for s in output_file_steps:
            meta.image_create_disk('coherence', s)

    @staticmethod
    def save_to_disk(meta, output_file_steps=''):

        if not output_file_steps:
            meta_info = meta.processes['coherence']
            output_file_keys = [key for key in meta_info.keys() if key.endswith('_output_file')]
            output_file_steps = [filename[:-13] for filename in output_file_keys]

        for s in output_file_steps:
            meta.image_memory_to_disk('coherence', s)
