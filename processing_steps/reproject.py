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

from rippl.resampling.coor_new_extend import CoorNewExtend
from rippl.resampling.grid_transforms import GridTransforms


class ProcessTemplate(Process):  # Change this name to the one of your processing step.

    def __init__(self, data_id='', coor_in=[], coor_out=[], slave=[], conversion_type='multilook'):

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

        :param CoordinateSystem coor_in: Coordinate system of the input grids.
        :param CoordinateSystem coor_out: Coordinate system of output grids, if not defined the same as coor_in

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        """

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """
        if conversion_type not in ['multilook', 'resample']:
            raise ValueError('Choose either multilook or resample as convesion type')

        self.conversion_type = conversion_type
        self.process_name = 'reproject'
        file_types = ['lines', 'pixels']
        data_types = ['real4', 'real4']

        # If the boundaries of the output grid are not jet defined, define them.
        if coor_out.shape == [0, 0]:
            coor_out = CoorNewExtend(coor_in, coor_out)

        # There are no input grids so loading of inputs is not needed.
        # Only thing that is calculated is how one grid projects onto another
        super(ProcessTemplate, self).__init__(
            process_name=self.process_name,
            data_id=data_id,
            file_types=file_types,
            process_dtypes=data_types,
            coor_in=coor_in,
            coor_out=coor_out,
            slave=slave)

    def process_calculations(self):
        """
        This step contains all the calculations that are done for this step. This is therefore the step that is most
        specific for every process definition.
        To run the full processing you have to initialize (__init__) and call (__call__) the object.

        To access the input data and save the output data you can the self.images dictionary. This contains all the
        memory data files. The keys are the same as the file_types and in_file_types.

        :return:
        """

        if
            transform = GridTransforms(self.coor_in, self.coor_out)

