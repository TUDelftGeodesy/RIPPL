"""
This function combines the deramping and resampling of a radar grid. This prevents writing data to disk after deramping
Basically this combines two other processing steps:
- deramp
- resample_radar_grid

"""

# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.processing_steps.deramp import Deramp
from rippl.resampling.resample_regular2irregular import Regural2irregular


class DerampResampleRadarGrid(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', in_coor=[], out_coor=[], slave='slave', overwrite=False, resample_type='4p_cubic',):
        """
        This function deramps the ramped data from TOPS mode to a deramped data. Input data of this function should
        be a radar coordinates grid.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs
        :param CoordinateSystem out_coor: Coordinate system of the input grids.
        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'resample'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['resampled']
        self.output_info['data_types'] = ['complex_short']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['slave', 'slave', 'slave']
        self.input_info['process_types'] = ['crop', 'geometric_coregistration', 'geometric_coregistration']
        self.input_info['file_types'] = ['crop', 'coreg_lines', 'coreg_pixels']
        self.input_info['polarisations'] = [polarisation, '', '']
        self.input_info['data_ids'] = [data_id, data_id, data_id]
        self.input_info['coor_types'] = ['in_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '']
        self.input_info['type_names'] = ['ramped_data', 'coreg_lines', 'coreg_pixels']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['in_coor'] = in_coor
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['resample_type'] = resample_type
        self.deramped = []
        self.settings['buf'] = 4
        self.settings['out_irregular_grids'] = ['coreg_lines', 'coreg_pixels']

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(DerampResampleRadarGrid, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self, test=False):
        """
        Main calculations done are based on information given in the meta data. The ramp can be calculated based on the
        DC and FM polynomials combined with range and azimuth time.

        Reference to the algorithm can be found here:
        https://earth.esa.int/documents/247904/1653442/Sentinel-1-TOPS-SLC_Deramping

        :return:
        """

        processing_data = self.processing_images['slave']
        if not isinstance(processing_data, ImageProcessingData):
            print('Input data missing')

        coordinates = self.coordinate_systems['in_block_coor']
        readfile = processing_data.readfiles['original']
        orbit = processing_data.find_best_orbit('original')
        coordinates.load_readfile(readfile)

        # Calculate azimuth/range grid and ramp.
        az_grid, ra_grid = Deramp.az_ra_time(coordinates)
        ramp = Deramp.calc_ramp(readfile, orbit, az_grid, ra_grid)

        # Finally calced the deramped image.
        self.deramped = self['ramped_data'] * ramp

        # Now apply the resampling
        # Init resampling
        resample = Regural2irregular(self.settings['resample_type'])

        in_coor = self.coordinate_systems['in_coor']
        out_coor = self.coordinate_systems['out_coor']
        in_block_coor = self.coordinate_systems['in_block_coor']

        # Change line/pixel coordinates to right value
        lines = (self['coreg_lines'] - in_block_coor.first_line) / \
                               (self.block_coor.multilook[0] / self.block_coor.oversample[0])
        pixels = (self['coreg_pixels'] - in_block_coor.first_pixel) / \
                               (self.block_coor.multilook[1] / self.block_coor.oversample[1])

        self['resampled'] = resample(self.deramped, lines, pixels)

    def test_result(self):
        """
        Method to check the results of the deramping. Here we generate the spectrogram of the input and output and
        create two images.

        :return:
        """

        # import needed function
        import matplotlib.pyplot as plt

        # Calculate the spectrogram of the ramped data
        spec_in = np.log(np.abs(np.fft.ifftshift(np.fft.fft2(self['ramped_data']))**2))

        # And the deramped data
        spec_out = np.log(np.abs(np.fft.ifftshift(np.fft.fft2(self.deramped)) ** 2))

        # plot in one image
        plt.figure()
        plt.subplot(211)
        plt.imshow(spec_in[:, ::10])

        plt.subplot(212)
        plt.imshow(spec_out[:, ::10])
        plt.show()
