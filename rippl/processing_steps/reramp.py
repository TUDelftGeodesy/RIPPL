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
from rippl.processing_steps.deramp import Deramp


class Reramp(Process):  # Change this name to the one of your processing step.

    def __init__(self, polarisation, data_id='', out_coor=[],
                 secondary_slc='secondary_slc', ramp_out=False, overwrite=False):
        """
        This function deramps the ramped data from TOPS mode to a deramped data. Input data of this function should
        be a radar coordinates grid.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param ImageProcessingData secondary_slc: Secondary image, used as the default for input and output for processing.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'reramp'
        self.output_info['image_type'] = 'secondary_slc'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = ['reramped']
        self.output_info['data_types'] = ['complex32']
        if ramp_out:
            self.output_info['file_names'].append('ramp')
            self.output_info['data_types'].append('float32')

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['secondary_slc', 'secondary_slc', 'secondary_slc']
        self.input_info['process_names'] = ['geometric_coregistration', 'geometric_coregistration', 'resample']
        self.input_info['file_names'] = ['coreg_lines', 'coreg_pixels', 'resampled']
        self.input_info['polarisations'] = ['', '', polarisation]
        self.input_info['data_ids'] = [data_id, data_id, data_id]
        self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '']
        self.input_info['aliases_processing'] = ['lines', 'pixels', 'resampled']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['secondary_slc'] = secondary_slc

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['ramp_out'] = ramp_out

    def process_calculations(self, test=False):
        """
        Main calculations done are based on information given in the metadata. The ramp can be calculated based on the
        DC and FM polynomials combined with range and azimuth time.

        Reference to the algorithm can be found here:
        https://earth.esa.int/documents/247904/1653442/Sentinel-1-TOPS-SLC_Deramping

        :return:
        """

        processing_data = self.processing_images['secondary_slc']
        if not isinstance(processing_data, ImageProcessingData):
            logging.info('Input data missing')

        readfile = processing_data.readfiles['original']
        orbit = processing_data.find_best_orbit('original')

        # Calculate azimuth/range grid and ramp.
        az_grid = self['lines'] * readfile.az_time_step + readfile.az_first_pix_time
        ra_grid = self['pixels'] * readfile.ra_time_step + readfile.ra_first_pix_time
        ramp = Deramp.calc_ramp(readfile, orbit, az_grid, ra_grid)

        # Finally calced the deramped image.
        self['reramped'] = self['resampled'] * np.conjugate(ramp)
        if self.settings['ramp_out']:
            self['ramp'] = np.angle(ramp).astype(np.float32)

    def test_result(self):
        """
        Method to check the results of the deramping. Here we generate the spectrogram of the input and output and
        create two images.

        :return:
        """

        # import needed function
        import matplotlib.pyplot as plt

        # Calculate the spectrogram of the ramped data
        spec_in = np.log(np.abs(np.fft.ifftshift(np.fft.fft2(self['input_data']))**2))

        # And the deramped data
        spec_out = np.log(np.abs(np.fft.ifftshift(np.fft.fft2(self['earth_topo_phase_corrected'])) ** 2))

        # plot in one image
        plt.figure()
        plt.subplot(211)
        plt.imshow(spec_in[:, ::10])

        plt.subplot(212)
        plt.imshow(spec_out[:, ::10])
        plt.show()
