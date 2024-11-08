'''
This is a parent class for different types of images like SLC and interferogram
'''
import os
import logging

from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.resampling.coor_concatenate import CoorConcatenate
from rippl.orbit_geometry.coordinate_system import CoordinateSystem


class ImageConcatData(ImageProcessingData):

    """
    :type data = ImageData
    :type slice_list = List(ImageProcessingData)
    """

    def __init__(self, folder, slice_list='', update_full_image=False, json_path='',
                 image_processing_meta='', overwrite=False):
        # Either give a xml_file or a res_file as input to define the meta_data of the image
        # Every image will contain slices (or bursts). This could be one or multiple.
        # Processing will be done based on these slices. Although some processes for which not everything needs to be
        # read in memory can be done seperately.

        # Read the image information
        super().__init__(folder=folder, overwrite=overwrite, json_path=json_path, image_processing_meta=image_processing_meta)
        # Adjust date to get the right values for dates and seconds in case of a date boundary.

        # Create the list of burst of this image
        self.slice_data = dict()

        # Read the individual slices ('slice_' + str(number) + '_swath_' + str(swath_no) + '_' + pol)
        # Length of these folders should be 20 characters
        self.slice_folders = next(os.walk(self.folder))[1]
        self.slice_names = sorted(['slice' + x.split('slice')[-1] for x in self.slice_folders if 'slice' in x])

        # If a slice list is defined remove the slices which are not included.
        if slice_list:
            self.slice_names = list(set(self.slice_names) & set(slice_list))
        self.slice_folders = [os.path.join(self.folder, os.path.basename(self.folder) + '_' + x) for x in self.slice_names]

        # For backward compatebility
        if not os.path.exists(self.json_path) or update_full_image:
            self.load_slice_meta()
            self.create_concatenate_meta_data()
            self.slice_data = dict()

    def processing_slice_image_data_exists(self, process, coordinates, in_coordinates, data_id, polarisation, process_type,
                                 slice='', disk_data=True):
        # Check if a certain file exists.

        if slice in self.slice_names:
            if len(self.slice_data.keys()) == 0:
                self.load_slice_meta()
            return self.slice_data[slice].processing_image_data_exists(process, coordinates, in_coordinates, data_id,
                                                                       polarisation, process_type, disk_data)
        else:
            logging.info('This slice name does not exist')

    def processing_slice_image_data_iterator(self, processes=[], coordinates=[], in_coordinates=[], data_ids=[],
                                             polarisations=[], file_types=[], data=True, load_memmap=True):
        """
        From the full_image and slices images are selected that fullfill the requirements given as an input.

        :param list[str] processes:
        :param list[CoordinateSystem] coordinates:
        :param in_coordinates:
        :param data_ids:
        :param polarisations:
        :param file_types:
        :param data:
        :return:
        """

        process_ids_out = []
        processes_out = []
        file_types_out = []
        coordinates_out = []
        in_coordinates_out = []
        slice_names_out = []
        images_out = []

        if len(self.slice_data.keys()) == 0:
            self.load_slice_meta()
        if data and load_memmap:
            if len(processes) == 0 and len(file_types) == 0:
                logging.info('Loading all memmap images of slices without defining processes or file_types could result '
                      'in reaching the limit of maximum open memmap files! Consider loading with defined processes '
                      'or file types to prevent this.')
            self.load_slice_memmap(processes=processes, file_types=file_types)

        for slice_name in self.slice_data.keys():
            processes_slice, process_ids_slice, coordinates_slice, in_coordinates_slice, file_types_slice, images_slice = \
                self.slice_data[slice_name].processing_image_data_iterator(processes, coordinates, in_coordinates, data_ids, polarisations, file_types)
            process_ids_out += process_ids_slice
            file_types_out += file_types_slice
            coordinates_out += coordinates_slice
            in_coordinates_out += in_coordinates_slice
            processes_out += processes_slice
            slice_names_out += [slice_name for n in range(len(process_ids_slice))]
            if data:
                images_out += images_slice
        
        return slice_names_out, processes_out, process_ids_out, coordinates_out, in_coordinates_out, file_types_out, \
               images_out

    def load_meta(self):
        # Reload the meta data
        super().__init__(folder=self.folder)

    def load_slice_meta(self, reference_images=False):
        # Read the information for individual slices
        self.slice_json_paths = []
        for slice_folder, slice_name in zip(self.slice_folders, self.slice_names):
            self.slice_data[slice_name] = ImageProcessingData(slice_folder)
            self.slice_json_paths.append(self.slice_data[slice_name].json_path)

    def load_slice_memmap(self, reference_images=False, processes=[], file_types=[]):
        # Read slices including memmap files
        self.load_slice_meta()
        for slice_name in self.slice_names:
            self.slice_data[slice_name].load_memmap_files(processes, file_types)

    def remove_slice_memmap(self, reference_images=False, processes=[], file_types=[]):
        # Remove memmap information
        for slice_name in self.slice_names:
            self.slice_data[slice_name].remove_memmap_files(processes, file_types)

    def remove_slice_memory(self, reference_images=False, processes=[], file_types=[]):
        # Remove memmap information
        for slice_name in self.slice_names:
            self.slice_data[slice_name].remove_memory_files(processes, file_types)

    def create_concatenate_meta_data(self):
        # This function creates a new concatenated image based on the crop information of the slices.

        # Create coordinate systems based on the readfiles.
        coors = []
        for slice in self.slice_names:
            coor = CoordinateSystem()
            coor.create_radar_coordinates()

            readfile_names = list(self.slice_data[slice].readfiles.keys())
            orbit_names = list(self.slice_data[slice].orbits.keys())
            orbit_prefix = [key.split('_')[0] for key in orbit_names]

            if 'original' in readfile_names and 'original' in orbit_prefix:
                coor.load_readfile(self.slice_data[slice].readfiles['original'])
                coor.load_orbit(self.slice_data[slice].orbits[orbit_names[orbit_prefix.index('original')]])
            elif 'reference' in readfile_names and 'reference' in orbit_prefix:
                coor.load_readfile(self.slice_data[slice].readfiles['reference'])
                coor.load_orbit(self.slice_data[slice].orbits[orbit_names[orbit_prefix.index('reference')]])
            else:
                coor.load_readfile(self.slice_data[slice].readfiles[readfile_names[0]])
                coor.load_orbit(self.slice_data[slice].orbits[orbit_names[0]])

            coors.append(coor)

        concat = CoorConcatenate(coors)
        concat.update_readfiles(coor.readfile, coor.orbit)

        concat_image = ImageProcessingData('')
        concat_image.add_readfile(concat.readfile)
        concat_image.add_orbit(coor.orbit)
        concat_image.create_header()
        folder = os.path.dirname(self.slice_data[slice].folder)
        concat_image.path = os.path.join(folder, os.path.basename(folder) + '.json')
        concat_image.save_json(concat_image.path)

        self.data = concat_image

    def create_concatenate_image(self, process, file_type, coor, data_id='', polarisation='', overwrite=False,
                                 output_type='disk', transition_type='full_weight', replace=False, cut_off=10,
                                 remove_input=False, tmp_directory=''):
        """
        This method is used to concatenate slices. Be sure that before this step is run the metadata is first created
        using the create_concatenate_meta_data function.

        :param str process: The process of which the result should be concatenated
        :param str file_type: Actual file type of the process that will be concatenated
        :param CoordinateSystem coor: Coordinatesystem of image. If size is already defined these will be used,
                    otherwise it will be calculated.
        :param str data_id: Data ID of process/file_type. Normally left empty
        :param str polarisation: Polarisation of data set
        :param bool overwrite: If data already exist, should we overwrite?
        :param str output_type: This is either memory or disk (Generally disk is preferred unless this dataset is not
                    saved to disk and is part of a processing pipeline.)
        :param str transition_type: Type of transition between burst. There are 3 types possible: 1) full weight, this
                    simply adds all values on top of each other. 2) linear, creates a linear transition zone between
                    the bursts. 3) cut_off, this creates a hard break between the different bursts and swaths without
                    overlap. (Note that option 2 and 3 are only possible when working in radar coordinates!)
        :param int cut_off: Number of pixels of the outer part of the image that will not be used because it could still
                    contain zeros.
        :param bool remove_input: Remove the disk data of the input images.
        :return:
        """

        # We do the import here to prevent a circular reference.
        from rippl.meta_data.concatenate import Concatenate

        # Initialize the concatenate step
        concatenate = Concatenate(self, process, file_type, coor, data_id, polarisation, overwrite, output_type)
        # Load data from slices
        concatenate.load_data()
        # Create output data for full image
        succes = concatenate.create_image(tmp_directory=tmp_directory)
        # Do the actual concatenation
        if succes:
            concatenate.concatenate(transition_type=transition_type, cut_off=cut_off, replace=replace, remove_input=remove_input)
        # Remove all memmap files after concatenation
        self.remove_memmap_files()
        self.remove_slice_memmap()
