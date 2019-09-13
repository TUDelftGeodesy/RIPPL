'''

This is a super class for all processing step. This class manages:
- Importing data from a processing stack (either SLC or interferogram)
- Initialize a processing data/metadata object.
- Writing results to disk if needed
- Removing results/ins from memory

'''

import numpy as np
import copy

from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.meta_data.process_data import ProcessData
from rippl.meta_data.image_data import ImageData
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.resampling.select_input_window import SelectInputWindow


class Process():

    def __init__(self, process_name='', data_id='', polarisation='', file_types=[], process_dtypes=[], shapes=[], settings=dict(),
                 coor_in=[], coor_out=[], in_coor_types=[], coordinate_systems=dict(),
                 in_image_types=[], in_processes=[], in_file_types=[], in_polarisations=[], in_data_ids=[], in_type_names = [],
                 slave=[], master=[], coreg_master=[], ifg=[], processing_images=dict(), out_processing_image='slave'):
        """
        Define the name and output files of the processing step.

        :param str process_name: Name of the process we are creating
        :param str data_id: Data ID of image. Only used in specific cases where the processing chain contains 2 times
                    the same process.
        :param str polarisation: Polarisation of processing outputs
        :param list[str] file_types: Names of the different outputs of the specific process
        :param list[str] process_dtypes: List of the process dtypes as defined in the ImageData class
        :param dict[str or float or int] settings: Any additional settings for the processing step. This variable is
                    used to store this information as meta data.

        :param CoordinateSystem coor_in: Coordinate system of the input grids.
        :param CoordinateSystem coor_out: Coordinate system of output grids, if not defined the same as coor_in
        :param list[str] in_coor_types: The coordinate types of the input grids. Sometimes some of these grids are
                different from the defined coor_in. Options are 'coor_in', 'coor_out' or anything else defined in the
                coordinate system input.
        :param dict[CoordinateSystem] coordinate_systems: Here the alternative input coordinate systems can be defined.
                Only used in very specific cases.

        :param list[str] in_image_types: The type of the input ImageProcessingData objects (e.g. slave/master/ifg etc.)
        :param list[str] in_processes: Which process outputs are we using as an input
        :param list[str] in_file_types: What are the exact outputs we use from these processes
        :param list[str] in_polarisations: For which polarisation is it done. Leave empty if not relevant
        :param list[str] in_data_ids: If processes are used multiple times in different parts of the processing they can be
                distinguished using an data_id. If this file_types, data_types, shapesis the case give the correct data_id. Leave empty if not relevant
        :param list[str] in_type_names: Names of the datasets used late in processing.

        :param ImageProcessingData slave: Slave image, used as the default for input and output for processing.
        :param ImageProcessingData master: Master image, generally used when creating interferograms
        :param ImageProcessingData coreg_master: Image used to coregister the slave image for resampline etc.
        :param ImageProcessingData ifg: Interferogram of a master/slave combination
        :param dict[ImageProcessingData] processing_images: Include other images than the slave/master/coreg_master/ifg 
                types. Only needed in very specific cases.
        :param str out_processing_image: Define which of the images is the output. Defaults to 'slave' if not given.
        """

        # Predefine all image definitions
        self.process_name = process_name    # type: str
        self.process = []                   # type: ProcessData

        # Coordinate systems used.
        self.coordinate_systems = dict()    # type: dict(CoordinateSystem)
        # Block coordinate system is the coordinate system create for processing in blocks.
        self.block_coor = []                # type: CoordinateSystem

        # Input and output image (slave/master/ifg etc.)
        self.in_processing_images = dict()  # type: dict(ImageProcessingData)
        self.out_processing_image = ''      # type: str

        # Input and output image data. To load from disk to memory and save results
        self.in_type_names = in_type_names  # type: list
        self.in_images = dict()             # type: dict(ImageData)
        self.out_images = dict()            # type: dict(ImageData)

        # Initialize the different input/output images. These steps can be done later in the processing and inputs can
        # therefore be left blank in initialization.
        self.define_coordinate_systems(coor_in, coor_out, coordinate_systems)
        self.define_processing_images(slave, master, coreg_master, ifg, processing_images, out_processing_image)

        # Create the process metadata and images. Load the input images too.
        self.create_process_metadata(polarisation, data_id, settings)
        self.create_output_images(file_types, process_dtypes, shapes)
        self.load_input_images(in_image_types, in_processes, in_file_types, in_polarisations, in_data_ids, in_coor_types,
                               in_type_names)        # We load the input images only. The data is not loaded to memory yet.

        # Information for processing of dataset in blocks. If you want to do so, run the define_block method.
        self.s_lin = 0
        self.s_pix = 0
        self.lines = self.coordinate_systems['coor_out'].shape[0]
        self.pixels = self.coordinate_systems['coor_out'].shape[1]
        self.out_irregular_grids = []
        self.in_irregular_grids = []
        self.blocks = False

    def __call__(self):
        """
        This function does basically the same as the __call__ function, but assumes that we apply no multiprocessing
        and or pipeline processing. Therefore it includes an extended number of steps:
        1. Reading in of input data
        2. Creation of output data on disk
        3. Create output memory files
        4. Perform the actual processing
        5. Save data to disk
        6. Clean memory steps.

        :return:
        """

        self.load_input_data()
        self.create_output_data_files()
        self.create_memory()
        self.process_calculations()
        self.save_to_disk()
        self.clean_memory()
        self.out_processing_image.update_json()

    def __getitem__(self, key):
        # Check if there is a memory file with this name and give the output.
        if key in list(self.in_images.keys()):
            data = self.in_images[key].memory['data']
            return data
        elif key in list(self.out_images.keys()):
            data = self.out_images[key].memory['data']
            return data
        else:
            return False

    def __setitem__(self, key, data):
        # Set the data of one variable in memory.
        if key in list(self.in_images.keys()):
            self.in_images[key].memory['data'] = data
            return True
        elif key in list(self.out_images.keys()):
            self.out_images[key].memory['data'] = data
            return True
        else:
            return False

    def process_calculations(self):
        """
        This is the function in every processing step where the actual calculations are done. Because this is the
        master function this method is empty, as there should be an override of this function in the child functions.

        :return:
        """

        print('This method should be in your process function to make any sense. For now all the outputs will just '
              'be filled with zeros!')

    def create_process_metadata(self, polarisation, data_id, settings):
        """
        Create the process data object. This creates the meta data of the image and creates a link to the output data
        on disk and in memory.

        :param str polarisation: Polarisation of data
        :param str data_id: data_id of dataset, if not relevant leave blank
        :param str settings: Specific settings of this processing step
        :return:
        """

        self.process_data = ProcessData(self.out_processing_image.folder, self.process_name, self.coor_out,
                                        settings, polarisation=polarisation, data_id=data_id)
        self.out_processing_image.add_process(self.process_data)
        self.process = self.process_data.meta

    def create_output_images(self, file_types, data_types, shapes):
        """
        This step is used to generate the output files of this function. These have to be defined in the initialization
        of a processing step. Best practice is to use a default set of inputs, but leave it to the user to define
        other input steps.

        :param list(str) file_types: Names of the file types to be generated.
        :param list(str) data_types: Data types of these files. Check the ImageData file for possible data formats
        :param list(tuple) shapes: Shapes of outputs. Defaults to shape of output coordinate system if not defined.
        :return:
        """

        # Add image data.
        self.process_data.add_process_images(file_types, data_types, shapes)
        # Define the output images.
        self.out_images = self.process_data.images

    def define_processing_images(self, slave='', master='', coreg_master='', ifg='', processing_images='',
                                 out_processing_image='slave'):
        """
        This function creates a list of the different images. A lot of functions will only need a slave image (for
        example deramping, resampling or multilooking). Other functions will need a coregistration master too (for
        example coregistration). Creating interferograms needs a slave and master image, but preferably also the coreg
        master.
        If for some reason other types of images are needed. These can be defined in the processing_images dictionary.
        Note that the naming of the images is just to make the distinction during processinge. e.g. an image can be
        a slave image in some cases and a master or coreg_master image in other cases.

        :param ImageProcessingData slave: Slave image
        :param ImageProcessingData master: Master image
        :param ImageProcessingData coreg_master: Image where the slave image is coregistered too.
        :param ImageProcessingData ifg: Interferogram image
        :param dict[ImageProcessingData] processing_images: Other images. Generally not needed.
        :return:
        """

        # Get the in images and rearrange
        self.in_processing_images = dict()
        if isinstance(slave, ImageProcessingData):
            self.in_processing_images['slave'] = slave
        if isinstance(master, ImageProcessingData):
            self.in_processing_images['master'] = master
        if isinstance(coreg_master, ImageProcessingData):
            self.in_processing_images['coreg_master'] = coreg_master
        if isinstance(ifg, ImageProcessingData):
            self.in_processing_images['ifg'] = ifg
        for image_key in processing_images.keys():
            if isinstance(processing_images[image_key], ImageProcessingData):
                self.in_processing_images[image_key] = processing_images[image_key]
        self.out_processing_image = self.in_processing_images[out_processing_image]

    def define_coordinate_systems(self, coor_in='', coor_out='', coordinate_systems=''):
        """
        This function defines a list of coordinate systems relevant for processing. When no multilooking or resampling
        is involved the coordinate system is the same for all input images. If not, also an output coordinate system
        is defined. In very rare cases where even a third coordinate system is involved also these coordinate systems
        can be loaded using the coordinate_systems variable.

        :param CoordinateSystem coor_in: Input coordinate system (if output coordinate system is left blank, output
                    coordinate system will be the same.
        :param CoordinateSystem coor_out: Output coordinate system
        :param dict[CoordinateSystem] coordinate_systems: Other coordinate systems used in this process.
        :return:
        """

        if isinstance(coor_in, CoordinateSystem):
            self.coor_in = coor_in
        else:
            print('input should be a CoordinateSystem object')
            return
        if not coor_out:
            self.coor_out = coor_in
        else:
            if isinstance(coor_out, CoordinateSystem):
                self.coor_out = coor_out
            else:
                print('input should be a CoordinateSystem object')
                return

        # Create a list of coordinate systems. Generally only coor_in and sometimes also coor_out is used. However, in
        # specific cases a third or fourth coordinate system can be added...
        self.coordinate_systems = coordinate_systems
        self.coordinate_systems['coor_in'] = self.coor_in
        self.coordinate_systems['coor_out'] = self.coor_out
        # Define the block coordinate system the same as the output coordinate system. In first instance they are the
        # same but if you define a processing block they will change. By default therefore the whole image is processed.
        self.block_coor = copy.deepcopy(self.coor_out)

    def define_processing_block(self, s_lin=0, s_pix=0, lines=0, pixels=0, in_irregular_grids=[], out_irregular_grids=[]):
        """
        Here we check whether the processing blocks can be processed based on the given sizes. Also we check whether
        it is possible to do block processing
        If the in coordinates and output coordinates are not the same either an irregular in or output grid is
        needed to do the reading in.
        Exception for this are the 'regular' multilooking, which does not need any precreated grid.

        :param int s_lin: Start line with respect too total image size
        :param int s_pix: Start pixel with respect too total image size
        :param int lines: Number of lines to process
        :param int pixels: Number of pixels to process
        :param in_irregular_grids: If input and output coordinates are not the same and conversion is irregular, the
                    irregular coordinates of the input expressed in the coordinates of the output grid.
        :param out_irregular_grids: If input and output coordinates are not the same and conversion is irregular, the
                    irregular coordinates of the output expressed in the coordinates of the input grid.
        :return:
        """

        self.s_lin = s_lin
        self.s_pix = s_pix
        self.lines = lines
        self.pixels = pixels
        self.blocks = True

        # Check the overlap and limit the number of lines if needed.
        if self.s_lin >= self.coor_out.shape[0] or self.s_pix >= self.coor_out.shape[1]:
            print('Start line and pixel are too high')
            return False
        if self.lines > (self.coor_out.shape[0] - self.s_lin) or self.lines == 0:
            self.lines = self.coor_out.shape[0] - self.s_lin
        if self.pixels > (self.coor_out.shape[1] - self.s_pix) or self.pixels == 0:
            self.pixels = self.coor_out.shape[1] - self.s_pix

        self.block_coor.first_line += self.s_lin
        self.block_coor.first_pixel += self.s_pix
        self.block_coor.shape = [self.lines, self.pixels]

        # Check if the input irregular/regular grid is given to select the right inputs.
        if not self.coor_in.same_coordinates(self.coor_out, strict=False):
            if len(list(self.in_images.keys())) == 0:
                # If there are no input images it is fine too.
                return True
            if isinstance(in_irregular_grids[0], ImageData):
                if in_irregular_grids[0].shape == self.coor_in.shape:
                    return True
            if isinstance(out_irregular_grids[0], ImageData):
                if out_irregular_grids[0].shape == self.coor_out.shape:
                    return True
            # in the case we have to change coordinate system the coordinates of the input/output grid have to be cal-
            # culated beforehand to apply the calculation.
            return False
        return True

    # Handling of in and output data. This is the main
    def load_input_images(self, image_types, processes, file_types, polarisations, data_ids, coor_types, type_names):
        """
        This function loads the input data needed to do the processing. To do so, the exact source of the input data 
        should be defined using the given variables.
        
        :param list[str] image_types: List of the image types we need (slave/master/ifg). If empty they are assumed to
                    be the same as the output image.
        :param list[str] processes: List of names of the processes
        :param list[str] file_types: List of file types, which are part of these processes.
        :param list[str] polarisations: List of polarisation of inputs. If not defined we select wat is available.
        :param list[str] data_ids: List of data ids. Only used when the same processing step is used more than once in
                    a processing chain. Generally empty.
        :param list[str] coor_types: The type of coordinate systems used. If not specified we assume that all inputs
                    are of the input coordinate system type.
        :return:
        """

        n_inputs = len(processes)
        if n_inputs == 0:
            return True

        # First check whether the image_types/coor_types/polarisations/data_ids are empty.
        if image_types == []:
            image_types = [self.out_processing_image for i in range(n_inputs)]
        else:
            check_image_types = [image_type in list(self.in_processing_images.keys()) for image_type in image_types]
            if False in check_image_types:
                TypeError('Specified image type does not exist in loaded input image types.')

        if coor_types == []:
            coor_types = ['coor_in' for i in range(n_inputs)]
        else:
            check_coor_types = [coor_type in list(self.coordinate_systems.keys()) for coor_type in coor_types]
            if False in check_coor_types:
                TypeError('Specified coordinate type does not exist in loaded coordinate systems.')

        if polarisations == []:
            polarisations = ['' for i in range(n_inputs)]
        if data_ids == []:
            data_ids = ['' for i in range(n_inputs)]

        # Check of they exist and get the images.
        for i, [image_type, process, file_type, polarisation, data_id, coor, name] in enumerate(zip(image_types,
                                                                                            processes,
                                                                                            file_types,
                                                                                            polarisations,
                                                                                            data_ids,
                                                                                            coor_types,
                                                                                            type_names)):
            image_data = self.in_processing_images[image_type].\
                processing_image_data_exists(process, self.coordinate_systems[coor], data_id, polarisation, file_type)
            if not image_data:
                return False
            else:
                self.in_images[name] = image_data

        return True

    def load_input_data(self, in_irregular_grids=[], out_irregular_grids=[], buf=3):
        """
        Here we actually load the needed input data. The images should already be loaded

        :param in_irregular_grids: Line and pixel grids with irregular input spacing (preferred for multilooking)
        :param out_irregular_grids: Line and pixel grids with irregular output spacing (preferred for resampling)
        :param int buf: The buffer we need for the correct resampling/multilooking of the data
        :return:
        """

        if self.blocks:
            if len(in_irregular_grids) == 2:
                s_lin, s_pix, shape = SelectInputWindow.input_irregular_rectangle(in_irregular_grids[0], in_irregular_grids[1],
                                                            self.s_lin, self.s_pix, [self.lines, self.pixels], buf)
            elif len(out_irregular_grids) == 2:
                s_lin, s_pix, shape = SelectInputWindow.output_irregular_rectangle(out_irregular_grids[0], out_irregular_grids[1],
                                                            self.s_lin, self.s_pix, [self.lines, self.pixels], buf)
            else:
                s_lin = self.s_lin
                s_pix = self.s_pix
                shape = [self.lines, self.pixels]
        else:
            s_lin = 0
            s_pix = 0
            shape = self.coor_in.shape

        for key in self.in_images.keys():
            if self.in_images[key].coordinates.id_str == self.coordinate_systems['coor_in'].id_str:
                self.in_images[key].load_memory_data(shape, s_lin, s_pix)
            elif self.in_images[key].coordinates.id_str == self.coordinate_systems['coor_out'].id_str:
                self.in_images[key].load_memory_data([self.lines, self.pixels], self.s_lin, self.s_pix)

    def create_memory(self, file_types=[]):
        """
        This function is used to create the outputs of the process to memory.

        :param list[str] file_types: File type of output grid
        :return:
        """

        if len(file_types) == 0:
            file_types = list(self.out_images.keys())

        for file_type in file_types:
            succes = self.out_images[file_type].new_memory_data([self.lines, self.pixels], self.s_lin, self.s_pix)

    # Next steps are not performed during the processing. These are executed before (create_output_data_file) or after
    # (save_to_disk, clean_memory) processing. However, they only need to be seperated when we work with data blocks.
    # Therefore, there is an added function beside __call__, called call_full, which does so.
    def create_output_data_files(self, file_types=[]):
        """
        This function preallocates the output files on disk. So it creates only files with zeros which can be filled
        later on in the process.

        :param list[str] file_types: File types to be preallocated on disk. If not specified, all output files will be
                        preallocated on disk.
        :return:
        """

        if len(file_types) == 0:
            file_types = list(self.out_images.keys())

        for file_type in file_types:
            create_new = self.out_images[file_type].create_disk_data()
            if not create_new:
                load_new = self.out_images[file_type].load_disk_data()
                if not load_new:
                    raise FileNotFoundError('Data on disk not found and not able to create')

        # If all files are succesfully loaded.
        return True

    def save_to_disk(self, file_types=[]):
        """
        This function saves all data that is stored in memory to disk. So during a processing step the data is first
        stored in memory and then saved to disk. This makes it possible to keep data in memory and use it for a proces
        sing pipeline. This function will therefore only be used as the last step of a pipeline.

        :param list(str) file_types: File types which are saved to disk. If not specified all file types are used.
        :return:
        """
        # Save data from memory to disk.

        if len(file_types) == 0:
            file_types = list(self.out_images.keys())

        for file_type in file_types:
            save_data = self.out_images[file_type].save_memory_data_to_disk()
            if not save_data:
                return False

        # If all files are succesfully saved.
        return True

    def clean_memory(self):
        """
        Clean all data from memory related to this processing step (in and output data). This is generally done after
        the processing of a single step or a processing pipeline is finished. This will prevent the use of to memory
        space for processing.

        :return:
        """

        for image in self.in_images.keys():
            self.in_images[image].remove_memory_data()
        for image in self.out_images.keys():
            self.out_images[image].remove_memory_data()
