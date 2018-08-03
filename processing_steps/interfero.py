# The following class creates an interferogram from a master and slave image.

from image_data import ImageData
from find_coordinates import FindCoordinates
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

    def __init__(self, master_meta, slave_meta, ifg_meta='', s_lin=0, s_pix=0, lines=0, multilook='', oversampling='', offset=''):
        # Add master image and slave if needed. If no slave image is given it should be done later using the add_slave
        # function.

        if isinstance(slave_meta, str):
            if len(slave_meta) != 0:
                self.slave = ImageData(slave_meta, 'single')
        elif isinstance(slave_meta, ImageData):
            self.slave = slave_meta
        if isinstance(master_meta, str):
            if len(master_meta) != 0:
                self.master = ImageData(master_meta, 'single')
        elif isinstance(master_meta, ImageData):
            self.master = master_meta
        if isinstance(ifg_meta, str):
            if len(ifg_meta) != 0:
                self.master = ImageData(ifg_meta, 'single')
            else:
                print('New interferogram created')
                self.create_meta_data(master_meta, slave_meta)
        elif isinstance(ifg_meta, ImageData):
            self.ifg = ifg_meta

        # If we did not define the shape (lines, pixels) of the file it will be done for the whole image crop
        self.multilook = []
        self.oversampling = []
        self.in_s_lin = []
        self.in_s_pix = []
        self.out_s_lin = []
        self.out_s_pix = []
        self.in_shapes = []
        self.out_shape = []
        self.offset = []
        self.sample = []

        if multilook == '':
            multilook = [[5, 20]]
        elif isinstance(multilook[0], int):
            multilook = [multilook]
        if offset == '':
            offset = [[20, 120]]
        elif isinstance(offset[0], int):
            offset = [offset]
        if oversampling == '':
            oversampling = [[1, 1]]
        elif isinstance(oversampling[0], int):
            oversampling = [oversampling]

        for ml, ovr, off in zip(multilook, oversampling, offset):

            sample_out, multilook_out, oversampling_out, offset_out, in_coor, out_coor = \
                FindCoordinates.multilook_coors(master=self.master, slave=self.slave, s_lin=s_lin, s_pix=s_pix, lines=lines, multilook=ml, oversampling=ovr, offset=off)

            self.offset.append(offset_out)
            self.sample.append(sample_out)
            self.multilook.append(multilook_out)
            self.oversampling.append(oversampling_out)
            self.in_s_lin.append(in_coor[0])
            self.in_s_pix.append(in_coor[1])
            self.in_shapes.append(in_coor[2])
            self.out_s_lin.append(out_coor[0])
            self.out_s_pix.append(out_coor[1])
            self.out_shape.append(out_coor[2])
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

        self.interferogram = dict()

    def __call__(self):
        # Check if needed data is loaded
        if len(self.slave_dat) == 0 or len(self.master_dat) == 0:
            print('Missing input data for creating interferogram for ' + self.ifg.folder + '. Aborting..')
            return False

        try:
            # Calculate the new line and pixel coordinates based orbits / geometry
            for sample, multilook, oversampling, min_pix, min_line, shape in \
                    zip(self.sample, self.multilook, self.oversampling, self.in_s_pix, self.in_s_lin, self.in_shapes):
                if multilook == [1, 1]:
                    self.interferogram[sample] = self.master_dat * self.slave_dat.conj()
                elif oversampling == [1, 1]:
                    ir = np.arange(0, shape[0], multilook[0])
                    jr = np.arange(0, shape[1], multilook[1])

                    ls = min_line - self.s_lin_min
                    ps = min_pix - self.s_pix_min
                    le = ls + shape[0]
                    pe = ps + shape[1]

                    self.interferogram[sample] = np.add.reduceat(np.add.reduceat(self.master_dat[ls:le, ps:pe] *
                                                                                 self.slave_dat[ls:le, ps:pe].conj(), ir), jr, axis=1)
                else:
                    ir_h = np.arange(0, shape[0], multilook[0] / oversampling[0])[:-(oversampling[0] - 1)]
                    ir = [[i, i + multilook[0]] for i in ir_h]
                    ir = [item for sublist in ir for item in sublist][:-1]
                    jr_h = np.arange(0, shape[1], multilook[1] / oversampling[1])[:-(oversampling[1] - 1)]
                    jr = [[i, i + multilook[1]] for i in jr_h]
                    jr = [item for sublist in jr for item in sublist][:-1]

                    ls = min_line - self.s_lin_min
                    ps = min_pix - self.s_pix_min
                    le = ls + shape[0]
                    pe = ps + shape[1]

                    self.interferogram[sample] = np.add.reduceat(np.add.reduceat(self.master_dat[ls:le, ps:pe] *
                                                                                 self.slave_dat[ls:le, ps:pe].conj(), ir)[::2, :], jr, axis=1)[:, ::2]

            # If needed do the multilooking step
            self.add_meta_data(self.master, self.slave, self.ifg, self.sample, self.out_shape, self.multilook, self.oversampling, self.offset)

            for out_s_lin, out_s_pix, sample in zip(self.out_s_lin, self.out_s_pix, self.sample):
                self.ifg.image_new_data_memory(self.interferogram[sample], 'interferogram', out_s_lin, out_s_pix, 'Data' + sample)

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

    def processed(self):
        # This function checks whether this step is already processed and exists on disk. That would

        print('Working on this')

    @staticmethod
    def create_meta_data(master, slave):
        # This function creates a folder and .res file for this interferogram.
        master_path = master.res_path
        slave_path = slave.res_path

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

    def create_output_files(self, to_disk=''):
        # Create the output files as memmap files for the whole image. If parallel processing is used this should be
        # done before the actual processing.

        if not to_disk:
            to_disk = ['Data']

        for s in to_disk:
            self.ifg.image_create_disk('interferogram', s)

    @staticmethod
    def add_meta_data(master, slave, ifg, sample_list, shape_list, multilook_list, oversampling_list, offset_list):
        # This function adds information about this step to the image. If parallel processing is used this should be
        # done before the actual processing.
        if 'interferogram' in ifg.processes.keys():
            meta_info = ifg.processes['interferogram']
        else:
            meta_info = OrderedDict()

        if 'coreg_readfiles' in master.processes.keys():
            meta_data = master
        elif 'coreg_readfiles' in slave.processes.keys():
            meta_data = slave
        else:
            print('Processing steps are missing in either master or slave image')
            return

        for sample, shape, multilook, oversample, offset in zip(sample_list, shape_list, multilook_list, oversampling_list, offset_list):
            dat = 'Data' + sample
            meta_info[dat + '_output_file'] = 'Ifg' + sample + '.raw'
            meta_info[dat + '_output_format'] = 'complex_real4'

            meta_info[dat + '_lines'] = str(shape[0])
            meta_info[dat + '_pixels'] = str(shape[1])
            meta_info[dat + '_first_line'] = meta_data.processes['earth_topo_phase']['Data_first_line']
            meta_info[dat + '_first_pixel'] = meta_data.processes['earth_topo_phase']['Data_first_pixel']
            meta_info[dat + '_multilook_azimuth'] = str(multilook[0])
            meta_info[dat + '_multilook_range'] = str(multilook[1])
            meta_info[dat + '_oversampling_azimuth'] = str(oversample[0])
            meta_info[dat + '_oversampling_range'] = str(oversample[1])
            meta_info[dat + '_offset_azimuth'] = str(offset[0])
            meta_info[dat + '_offset_range'] = str(offset[1])

        ifg.image_add_processing_step('interferogram', meta_info)



    @staticmethod
    def processing_info():
        # Information on this processinag step
        input_dat = defaultdict()
        input_dat['master']['crop', 'earth_topo_phase'] = ['Data']
        input_dat['slave']['crop', 'earth_topo_phase'] = ['Data']

        output_dat = dict()
        output_dat['ifgs']['interferogram'] = ['Data']

        # Number of times input data is used in ram. Bit difficult here but 5 times is ok guess.
        mem_use = 3

        return input_dat, output_dat, mem_use
