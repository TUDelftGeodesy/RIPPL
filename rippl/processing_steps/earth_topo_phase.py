# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np
import logging

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class EarthTopoPhase(Process):  # Change this name to the one of your processing step.

    def __init__(self, polarisation, data_id='', out_coor=[], secondary_slc='secondary_slc', overwrite=False,
                 earth_topo_phase_out=False, reramped=True):
        """
        This function deramps the ramped data from TOPS mode to a deramped data. Input data of this function should
        be a radar coordinates grid.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem out_coor: Coordinate system of the input grids.
        :param ImageProcessingData secondary_slc: Secondary image, used as the default for input and output for processing.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'earth_topo_phase'
        self.output_info['image_type'] = 'secondary_slc'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = ['earth_topo_phase_corrected']
        self.output_info['data_types'] = ['complex32']
        if earth_topo_phase_out:
            self.output_info['file_names'].append('earth_topo_phase')
            self.output_info['data_types'].append('float32')

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['secondary_slc', 'secondary_slc']
        if reramped:
            self.input_info['process_names'] = ['geometric_coregistration', 'reramp']
            self.input_info['file_names'] = ['coreg_pixels', 'reramped']
        else:
            self.input_info['process_names'] = ['geometric_coregistration', 'resample']
            self.input_info['file_names'] = ['coreg_pixels', 'resampled']

        self.input_info['polarisations'] = ['', polarisation]
        self.input_info['data_ids'] = [data_id, data_id]
        self.input_info['coor_types'] = ['out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '']
        self.input_info['aliases_processing'] = ['pixels', 'resampled']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['secondary_slc'] = secondary_slc

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['earth_topo_phase_out'] = earth_topo_phase_out

    def process_calculations(self):
        """
        Because of the change in baselines, every pixel is shifted in azimuth time. This also influences the delay time
        to the same point on the ground for different orbits. Therefore, we correct here for this effect using the
        geometrical shift in range.

        :return:
        """

        processing_data = self.processing_images['secondary_slc']
        if not isinstance(processing_data, ImageProcessingData):
            logging.info('Input data missing')

        readfile = processing_data.readfiles['original']
        orbit = processing_data.find_best_orbit('original')
        in_coor = self.in_images['pixels'].in_coordinates

        # Calculate azimuth/range grid and ramp.
        ra_in = np.tile(((np.arange(self.coordinate_systems['out_coor_chunk'].shape[1]) + in_coor.first_pixel) * in_coor.ra_step + in_coor.ra_time)
                        [None, :], (self.coordinate_systems['out_coor_chunk'].shape[0], 1))

        ra_shift = (self['pixels'] * self.coordinate_systems['out_coor_chunk'].ra_step) + self.coordinate_systems['out_coor_chunk'].ra_time - ra_in
        c = 299792458       # Speed of light m/s

        phase_data = (np.remainder(ra_shift * c / readfile.wavelength, 1) * np.pi * 2).astype(np.float32)
        if self.settings['earth_topo_phase_out']:
            self['earth_topo_phase'] = phase_data

        ramp = np.exp(-1j * phase_data).astype(np.complex64)

        # Finally remove the earth topographic phase
        self['earth_topo_phase_corrected'] = self['resampled'] * np.conjugate(ramp)
