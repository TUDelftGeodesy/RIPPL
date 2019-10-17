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
                 coordinates=[],
                 slave=[], master=[], ifg=[], out_processing_image='slave', overwrite=False):

        """
        In the template all options are still given. But for the final processing step it is possible to remove a part
        of the inputs to make it more clear what is needed as an input.
        The input parameters can be divided in 4 blocks:
        1. The first block defines the data_id and polarisation of this step. For functions where no actual radar data
                is processed, the polarisation value will be irrelevant.
        2. The second block is about the input coordinate systems. Many functions will need the in_coor variable only
                as there is no change in coordinate systems.
        3. The third block is about the input data. In general it is ok to leave it open and use defaults when nothing
                is given. It is also possible to force certain default names and leave for example only the
                in_polarisations and in_data_ids open.
        4. The last block is about the processing inputs that are used. In many cases only the definition of the slave
                or the coreg_master are ok.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem coordinates: Coordinate system of the input grids.

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData master: Master image, generally used when creating interferograms
        :param ImageProcessingData ifg: Interferogram of a master/slave combination
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'coherence'
        self.output_info['image_type'] = 'ifg'
        self.output_info['polarisation'] = polarisation
        self.output_info['data_id'] = data_id
        self.output_info['coor_type'] = 'out_coor'
        self.output_info['file_types'] = ['coherence']
        self.output_info['data_types'] = ['real4']

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['slave', 'master', 'ifg']
        self.input_info['process_types'] = ['square_amplitude', 'square_amplitude', 'interferogram']
        self.input_info['file_types'] = ['square_amplitude', 'square_amplitude', 'interferogram']
        self.input_info['polarisations'] = [polarisation, polarisation, polarisation]
        self.input_info['data_ids'] = [data_id, data_id, data_id]
        self.input_info['coor_types'] = ['out_coor', 'out_coor', 'out_coor']
        self.input_info['in_coor_types'] = ['', '', '']
        self.input_info['type_names'] = ['slave', 'master', 'ifg']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = coordinates
        self.coordinate_systems['in_coor'] = coordinates

        # image data processing
        self.processing_images = dict()
        self.processing_images['slave'] = slave
        self.processing_images['master'] = master
        self.processing_images['ifg'] = ifg

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()

    def init_super(self):

        super(Coherence, self).__init__(
            input_info=self.input_info,
            output_info=self.output_info,
            coordinate_systems=self.coordinate_systems,
            processing_images=self.processing_images,
            overwrite=self.overwrite,
            settings=self.settings)

    def process_calculations(self):
        """
        This step contains all the calculations that are done for this step. This is therefore the step that is most
        specific for every process definition.
        To run the full processing you have to initialize (__init__) and call (__call__) the object.

        To access the input data and save the output data you can the self.images dictionary. This contains all the
        memory data files. The keys are the same as the file_types and in_file_types.

        :return:
        """

        valid_id = (self['slave'] != 0) * (self['master'] != 0)
        self['coherence'][valid_id] = np.abs(self['ifg'][valid_id]) / np.sqrt(self['slave'][valid_id] * self['master'][valid_id])
