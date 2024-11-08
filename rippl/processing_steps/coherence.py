# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class Coherence(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', polarisation='',
                 out_coor=[],
                 secondary_slc='secondary_slc', primary_slc='primary_slc', ifg='ifg', out_processing_image='ifg', overwrite=False):

        """
        In the template all options are still given. But for the final processing step it is possible to remove a part
        of the inputs to make it more clear what is needed as an input.
        The input parameters can be divided in 4 chunks:
        1. The first chunk defines the data_id and polarisation of this step. For functions where no actual radar data
                is processed, the polarisation value will be irrelevant.
        2. The second chunk is about the input coordinate systems. Many functions will need the in_coor variable only
                as there is no change in coordinate systems.
        3. The third chunk is about the input data. In general it is ok to leave it open and use defaults when nothing
                is given. It is also possible to force certain default names and leave for example only the
                in_polarisations and in_data_ids open.
        4. The last chunk is about the processing inputs that are used. In many cases only the definition of the secondary_slc
                or the reference_slc are ok.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem out_coor: Coordinate system of the input grids.

        :param ImageProcessingData secondary_slc: Secondary image, used as the default for input and output for processing.
        :param ImageProcessingData primary_slc: primary image, generally used when creating interferograms
        :param ImageProcessingData ifg: Interferogram of a primary_slc/secondary_slc combination
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'coherence'
        self.output_info['image_type'] = 'ifg'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_names'] = ['coherence']
        self.output_info['data_types'] = ['real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['secondary_slc', 'primary_slc', 'ifg']
        self.input_info['process_names'] = ['intensity', 'intensity', 'interferogram']
        self.input_info['file_names'] = ['intensity', 'intensity', 'interferogram']
        self.input_info['polarisations'] = [polarisation, polarisation, polarisation]
        self.input_info['data_ids'] = [data_id, data_id, data_id]
        self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '']
        self.input_info['aliases_processing'] = ['secondary_slc', 'primary_slc', 'ifg']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['secondary_slc'] = secondary_slc
        self.processing_images['primary_slc'] = primary_slc
        self.processing_images['ifg'] = ifg

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

    def process_calculations(self):
        """
        This step contains all the calculations that are done for this step. This is therefore the step that is most
        specific for every process definition.
        To run the full processing you have to initialize (__init__) and call (__call__) the object.

        To access the input data and save the output data you can the self.images dictionary. This contains all the
        memory data files. The keys are the same as the file_types and in_file_types.

        :return:
        """

        valid_id = (self['secondary_slc'] != 0) * (self['primary_slc'] != 0) * (self['ifg'] != 0)
        self['coherence'][valid_id] = np.abs(self['ifg'][valid_id]) / np.sqrt(self['secondary_slc'][valid_id] * self['primary_slc'][valid_id])
