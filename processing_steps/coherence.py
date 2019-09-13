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
                 coor_in=[],
                 in_image_types=[], in_coor_types=[], in_processes=[], in_file_types=[], in_polarisations=[],
                 in_data_ids=[],
                 slave=[], master=[], ifg=[], out_processing_image='slave'):

        """
        In the template all options are still given. But for the final processing step it is possible to remove a part
        of the inputs to make it more clear what is needed as an input.
        The input parameters can be divided in 4 blocks:
        1. The first block defines the data_id and polarisation of this step. For functions where no actual radar data
                is processed, the polarisation value will be irrelevant.
        2. The second block is about the input coordinate systems. Many functions will need the coor_in variable only
                as there is no change in coordinate systems.
        3. The third block is about the input data. In general it is ok to leave it open and use defaults when nothing
                is given. It is also possible to force certain default names and leave for example only the
                in_polarisations and in_data_ids open.
        4. The last block is about the processing inputs that are used. In many cases only the definition of the slave
                or the coreg_master are ok.

        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs

        :param CoordinateSystem coor_in: Coordinate system of the input grids.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.)
        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_polarisations: For which polarisation is it done. Leave empty if not relevant
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData master: Master image, generally used when creating interferograms
        :param ImageProcessingData ifg: Interferogram of a master/slave combination
        """

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """

        self.process_name = 'coherence'
        file_types = ['coherence']
        data_types = ['real4']

        """
        Then give the default input steps for the processing. The given input values will be overridden when other input
        values are given.
        """
        if len(in_image_types) == 0:
            in_image_types = ['slave', 'master', 'ifg']
        if len(in_coor_types) == 0:
            in_coor_types = ['coor_in', 'coor_in', 'coor_in']
        if len(in_data_ids) == 0:
            in_data_ids = ['none', 'none', 'none']
        if len(in_polarisations) == 0:
            in_polarisations = ['', '', '']
        if len(in_processes) == 0:
            in_processes = ['square_amplitude', 'square_amplitude', 'interferogram']
        if len(in_file_types) == 0:
            in_file_types = ['square_amplitude', 'square_amplitude', 'interferogram']

        in_type_names = ['slave', 'master', 'ifg']

        # Initialize
        super(Coherence, self).__init__(
                       process_name=self.process_name,
                       data_id=data_id, polarisation=polarisation,
                       file_types=file_types,
                       process_dtypes=data_types,
                       coor_in=coor_in,
                       in_type_names=in_type_names,
                       in_coor_types=in_coor_types,
                       in_image_types=in_image_types,
                       in_processes=in_processes,
                       in_file_types=in_file_types,
                       in_polarisations=in_polarisations,
                       in_data_ids=in_data_ids,
                       slave=slave,
                       master=master,
                       ifg=ifg,
                       out_processing_image='ifg')

    def process_calculations(self):
        """
        This step contains all the calculations that are done for this step. This is therefore the step that is most
        specific for every process definition.
        To run the full processing you have to initialize (__init__) and call (__call__) the object.

        To access the input data and save the output data you can the self.images dictionary. This contains all the
        memory data files. The keys are the same as the file_types and in_file_types.

        :return:
        """

        self['coherence'] = np.abs(self['ifg']) / np.sqrt(self['slave'] * self['master'])
