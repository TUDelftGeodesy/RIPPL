# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class Intensity(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='', out_coor='', secondary_slc='secondary_slc', overwrite=False,
                 resampled=True):

        """
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param ImageProcessingData or str secondary_slc: Secondary image, used as the default for input and output for processing.
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'intensity'
        self.output_info['image_type'] = 'secondary_slc'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = ['intensity']
        self.output_info['data_types'] = ['real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['secondary_slc']
        if not resampled:
            self.input_info['process_names'] = ['crop']
            self.input_info['file_names'] = ['crop']
        else:
            self.input_info['process_names'] = ['earth_topo_phase']
            self.input_info['file_names'] = ['earth_topo_phase_corrected']

        self.input_info['polarisations'] = [polarisation]
        self.input_info['data_ids'] = [data_id]
        self.input_info['coor_types'] = ['out_coor']
        self.input_info['in_coor_types'] = ['']
        self.input_info['aliases_processing'] = ['complex_data']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['secondary_slc'] = secondary_slc

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

    def process_calculations(self):
        """
        Create a square amplitude image. (Very basic operation)

        :return:
        """

        self['intensity'] = np.abs(self['complex_data'])**2
