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


class EarthTopoPhase(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', out_coor=[], in_image_types=[], slave='slave', overwrite=False):
        """
        This function deramps the ramped data from TOPS mode to a deramped data. Input data of this function should
        be a radar coordinates grid.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.)

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'earth_topo_phase'
        self.output_info['image_type'] = 'slave'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['earth_topo_phase_corrected']
        self.output_info['data_types'] = ['complex_short']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['slave', 'slave']
        self.input_info['process_types'] = ['resample', 'geometric_coregistration']
        self.input_info['file_types'] = ['resampled', 'coreg_pixels']
        self.input_info['polarisations'] = [polarisation, '']
        self.input_info['data_ids'] = [data_id, '']
        self.input_info['coor_types'] = ['out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '']
        self.input_info['type_names'] = ['input_data', 'pixels']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

    def init_super(self):

        self.load_coordinate_system_sizes()
        super(EarthTopoPhase, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        Because of the change in baselines, every pixel is shifted in azimuth time. This also influences the delay time
        to the same point on the ground for different orbits. Therefore we correct here for this effect using the
        geometrical shift in range.

        :return:
        """

        processing_data = self.processing_images['slave']
        if not isinstance(processing_data, ImageProcessingData):
            print('Input data missing')

        readfile = processing_data.readfiles['original']
        orbit = processing_data.find_best_orbit('original')
        in_coor = self.in_images['pixels'].in_coordinates

        # Calculate azimuth/range grid and ramp.
        ra_in = np.tile(((np.arange(self.coordinate_systems['block_coor'].shape[1]) + in_coor.first_pixel) * in_coor.ra_step + in_coor.ra_time)
                        [None, :], (self.coordinate_systems['block_coor'].shape[0], 1))

        ra_shift = (self['pixels']  * self.coordinate_systems['block_coor'].ra_step) + self.coordinate_systems['block_coor'].ra_time - ra_in
        c = 299792458
        ramp = np.exp(-1j * (ra_shift * c / readfile.wavelength) * 2 * np.pi).astype(np.complex64)

        # Finally calced the deramped image.
        self['earth_topo_phase_corrected'] = self['input_data'] * np.conjugate(ramp)
