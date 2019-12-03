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


class ProcessTemplate(Process):     # Change this name to the one of your processing step.


    def __init__(self, data_id='', polarisation='',
                 in_coor=[], out_coor=[], coordinate_systems=dict(),
                 in_image_types=[], in_coor_types=[], in_processes=[], in_file_types=[], in_polarisations=[], in_data_ids=[],
                 slave='slave', master='master', coreg_master='coreg_master', ifg='ifg', processing_images=dict(), out_processing_image='slave',
                 overwrite=False):

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

        :param CoordinateSystem in_coor: Coordinate system of the input grids.
        :param CoordinateSystem out_coor: Coordinate system of output grids, if not defined the same as in_coor
        :param list[str] in_coor_types: The coordinate types of the input grids. Sometimes some of these grids are
                different from the defined in_coor. Options are 'in_coor', 'out_coor' or anything else defined in the
                coordinate system input.
        :param dict[CoordinateSystem] coordinate_systems: Here the alternative input coordinate systems can be defined.
                Only used in very specific cases.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.)
        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_polarisations: For which polarisation is it done. Leave empty if not relevant
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this is the case give the correct data_id. Leave empty if not relevant

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData master: Master image, generally used when creating interferograms
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        :param ImageProcessingData ifg: Interferogram of a master/slave combination
        :param dict[ImageProcessingData] processing_images: Include other images than the slave/master/coreg_master/ifg
                types. Only needed in very specific cases.
        :param str out_processing_image: Define which of the images is the output. Defaults to 'slave' if not given.
        """

        """
        First define the name and output types of this processing step.
        1. process_name > name of process
        2. file_types > name of process types that will be given as output
        3. data_types > names 
        """

        self.process_name = 'template'
        file_types = ['template_1', 'template_2']
        data_types = ['real4', 'real4']

        """
        Then give the default input steps for the processing. The given input values will be overridden when other input
        values are given.
        """
        if len(in_image_types) == 0:
            in_image_types = ['slave', 'slave']    # In this case only the slave is used, so the input variable master,
                                                   # coreg_master, ifg and processing_images are not needed.
                                                   # However, if you override the default values this could change.
        if len(in_coor_types) == 0:
            in_coor_types = ['in_coor', 'in_coor'] # Same here but then for the out_coor and coordinate_systems
        if len(in_data_ids) == 0:
            in_data_ids = ['none', 'none']
        if len(in_polarisations) == 0:
            in_polarisations = ['none', 'none']
        if len(in_processes) == 0:
            in_processes = ['crop', 'geocode']
        if len(in_file_types) == 0:
            in_file_types = ['crop', 'latitude']

        # The names for the inputs that will be used during the processing.
        in_type_names = ['input_data', 'lat']

        """
        If needed define the grids that represent the coordinates of the irregular input/output grids. This is only 
        needed when you want to do calculations in blocks.
        """

        """
        Finally initialize using the parent Process class. All the input data parameters that are not given in the 
        initialization of this function, can be left out here too.
        For example, if you use only one coordinate system you can leave out out_coor, in_coor_types and 
        coordinate_systems.
        
        In most cases most of the variables are not used, but we give them here to be complete. Check one of the basic
        processing steps like deramping or resampling for the normal need of those variables. 
        """
        super(ProcessTemplate, self).__init__(
                       process_name=self.process_name,
                       data_id=data_id, polarisation=polarisation,
                       file_types=file_types,
                       process_dtypes=data_types,
                       in_coor=in_coor,
                       out_coor=out_coor,
                       in_coor_types=in_coor_types,
                       coordinate_systems=coordinate_systems,
                       in_type_names=in_type_names,
                       in_image_types=in_image_types,
                       in_processes=in_processes,
                       in_file_types=in_file_types,
                       in_polarisations=in_polarisations,
                       in_data_ids=in_data_ids,
                       slave=slave,
                       master=master,
                       coreg_master=coreg_master,
                       ifg=ifg,
                       processing_images=processing_images,
                       out_processing_image=out_processing_image,
                       overwrite=overwrite)

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
