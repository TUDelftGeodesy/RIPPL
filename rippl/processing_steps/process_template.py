# This processing file is a template for other processing steps.
# Do not change the already given steps in this function to prevent problems with creating pipeline processing
# later on.

# Try to do all calculations using numpy functions.
import numpy as np

# Import the parent class Process for processing steps.
from rippl.meta_data.process import Process
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class ProcessTemplate(Process):     # Change this name to the one of your processing step.


    def __init__(self, data_id='', polarisation='',
                 in_coor=[], out_coor=[], coordinate_systems=dict(),
                 in_image_types=[], in_coor_types=[], in_processes=[], in_file_types=[], in_polarisations=[], in_data_ids=[],
                 secondary_slc='secondary_slc', primary_slc='primary_slc', reference_slc='reference_slc', ifg='ifg', processing_images=dict(), out_processing_image='secondary_slc',
                 overwrite=False):

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

        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor
        :param list[str] in_coor_types: The coordinate types of the input grids. Sometimes some of these grids are
                different from the defined in_coor. Options are 'in_coor', 'out_coor' or anything else defined in the
                coordinate system input.
        :param dict[CoordinateSystem] coordinate_systems: Here the alternative input coordinate systems can be defined.
                Only used in very specific cases.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. secondary_slc/primary_slc/ifg etc.)
        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_polarisations: For which polarisation is it done. Leave empty if not relevant
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData secondary_slc: Secondary image, used as the default for input and output for processing.
        :param ImageProcessingData primary_slc: primary image, generally used when creating interferograms
        :param ImageProcessingData reference_slc: Image used to coregister the secondary_slc image for resampline etc.
        :param ImageProcessingData ifg: Interferogram of a primary_slc/secondary_slc combination
        :param dict[ImageProcessingData] processing_images: Include other images than the secondary_slc/primary_slc/reference_slc/ifg
                types. Only needed in very specific cases.
        :param str out_processing_image: Define which of the images is the output. Defaults to 'secondary_slc' if not given.
        """

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """

        # Output data information
        self.output_info = dict()
        self.output_info['process_name'] = 'template'
        self.output_info['image_type'] = 'template'             # Use reference_slc, secondary_slc, primary_slc, ifg
        self.output_info['polarisation'] = ''                   # Use HH, HV, VV, VH
        self.output_info['data_id'] = data_id                   # Use if needed, otherwise empty ''
        self.output_info['coor_type'] = 'out_coor'              # Use out_coor
        self.output_info['file_names'] = ['output_variable_name_1', 'output_variable_name_2']    # Give the variable file names of the outputs
        self.output_info['data_types'] = ['float32', 'float32'] # Data type output, int16/int32/float16/float32/float64/complex64/complex_int/complex_short

        # Input data information
        self.input_info = dict()
        self.input_info['image_types'] = ['reference_slc', 'reference_slc']         # Use reference_slc, secondary_slc, primary_slc, ifg
        self.input_info['process_names'] = ['input_process_1', 'input_process_2']   # Give the input processing names
        self.input_info['file_names'] = ['variable_name_1', 'variable_name_2']      # Give the file/variable name of the input process
        self.input_info['polarisations'] = ['', '']                                 # Input polarisation if relevant
        self.input_info['data_ids'] = [data_id, data_id]                            # Input data_id if relevant
        self.input_info['coor_types'] = ['in_coor', 'in_coor']                      # Input coordinates
        self.input_info['in_coor_types'] = ['']
        self.input_info['aliases_processing'] = ['dem']

        # Coordinate systems
        self.coordinate_systems = dict()
        self.coordinate_systems['out_coor'] = out_coor
        self.coordinate_systems['in_coor'] = in_coor

        # image data processing
        self.processing_images = dict()
        self.processing_images['reference_slc'] = reference_slc

        # Finally define whether we overwrite or not
        self.overwrite = overwrite
        self.settings = dict()
        self.settings['dem_type'] = 'test'
        self.settings['in_coor'] = dict()
        self.settings['in_coor']['buffer'] = 0
        self.settings['in_coor']['rounding'] = 0
        self.settings['in_coor']['min_height'] = 0
        self.settings['in_coor']['max_height'] = 0

        """
        Then give the default input steps for the processing. The given input values will be overridden when other input
        values are given.
        """

        """
        If needed define the grids that represent the coordinates of the irregular input/output grids. This is only 
        needed when you want to do calculations in chunks.
        """

        """
        Finally initialize using the parent Process class. All the input data parameters that are not given in the 
        initialization of this function, can be left out here too.
        For example, if you use only one coordinate system you can leave out out_coor, in_coor_types and 
        coordinate_systems.
        
        In most cases most of the variables are not used, but we give them here to be complete. Check one of the basic
        processing steps like deramping or resampling for the normal need of those variables. 
        """

    def process_calculations(self):
        """
        This step contains all the calculations that are done for this step. This is therefore the step that is most
        specific for every process definition.
        To run the full processing you have to initialize (__init__) and call (__call__) the object.

        To access the input data and save the output data you can the self.images dictionary. This contains all the
        memory data files. The keys are the same as the file_types and in_file_types.

        :return:
        """

        pass
