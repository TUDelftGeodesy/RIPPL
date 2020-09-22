'''
This is a parent class for different types of images like SLC and interferogram
'''
import os

from rippl.meta_data.image_data import ImageData
from rippl.meta_data.image_processing_data import ImageProcessingData
from rippl.resampling.coor_concatenate import CoorConcatenate
from rippl.orbit_geometry.coordinate_system import CoordinateSystem
from rippl.meta_data.process_data import ProcessData, ProcessMeta


class ImageConcatData(object):

    """
    :type data = ImageData
    :type slice_list = List(ImageProcessingData)
    """

    def __init__(self, folder, slice_list='', update_full_image=False, json_path='', adjust_date=False):
        # Either give an xml_file or a res_file as input to define the meta_data of the image
        # Every image will contain slices (or bursts). This could be one or multiple.
        # Processing will be done based on these slices. Although some processes for which not everything needs to be
        # read in memory can be done seperately.

        # Read the image information
        self.folder = folder
        self.adjust_date = adjust_date

        if not json_path:
            self.json_path = os.path.join(self.folder, 'info.json')
        else:
            self.json_path = json_path
        self.data = ImageProcessingData(self.folder)
        # Adjust date to get the right values for dates and seconds in case of a date boundary.

        # Other images that are linked to this image (added in SLC or interferogram)
        self.reference_images = dict()

        # Create the list of burst of this image
        self.slice_data = dict()

        # Read the individual slices ('slice_' + str(number) + '_swath_' + str(swath_no) + '_' + pol)
        # Length of these folders should be 20 characters
        self.slice_folders = next(os.walk(self.folder))[1]
        self.slice_names = sorted([x for x in self.slice_folders if len(x) == 17])

        # If a slice list is defined remove the slices which are not included.
        if slice_list:
            self.slice_names = list(set(self.slice_names) & set(slice_list))
        self.slice_folders = [os.path.join(self.folder, x) for x in self.slice_names]

        if not os.path.exists(self.json_path) or update_full_image:
            self.load_slice_meta()
            self.create_concatenate_meta_data()
            self.slice_data = dict()

        self.load_full_meta()
        self.load_full_memmap()

    def concat_image_data_exists(self, process, coordinates, in_coordinates, data_id, polarisation, process_type,
                                 slice='', disk_data=True):
        # Check if a certain file exists.

        if slice in self.slice_names:
            if len(self.slice_data.keys()) == 0:
                self.load_slice_meta()
            return self.slice_data[slice].processing_image_data_exists(process, coordinates, in_coordinates, data_id,
                                                                       polarisation, process_type, disk_data)
        else:
            if not isinstance(self.data, ImageProcessingData):
                self.load_full_meta()
            return self.data.processing_image_data_exists(process, coordinates, in_coordinates, data_id, polarisation,
                                                                       process_type, disk_data)

    def concat_image_data_iterator(self, processes=[], coordinates=[], in_coordinates=[], data_ids=[], polarisations=[], file_types=[],
                                   full_image=True, slices=True, data=True):
        """
        From the full_image and slices images are selected that fullfill the requirements given as an input.

        :param list[str] processes:
        :param list[CoordinateSystem] coordinates:
        :param in_coordinates:
        :param data_ids:
        :param polarisations:
        :param file_types:
        :param full_image:
        :param slices:
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

        if full_image:
            if not isinstance(self.data, ImageProcessingData):
                self.load_full_meta()
            if data:
                self.load_full_memmap()

            processes_full, process_ids_full, coordinates_full, in_coordinates_full, file_types_full, images_full = \
                self.data.processing_image_data_iterator(processes, coordinates, in_coordinates, data_ids, polarisations, file_types)
            process_ids_out += process_ids_full
            file_types_out += file_types_full
            coordinates_out += coordinates_full
            in_coordinates_out += in_coordinates_full
            processes_out += processes_full
            slice_names_out += ['' for n in range(len(process_ids_full))]
            if data:
                images_out += images_full

        if slices:
            if len(self.slice_data.keys()) == 0:
                self.load_slice_meta()
            if data:
                self.load_slice_memmap()

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

    def reference_images_iterator(self, function_name, reference_images=True):
        # Run the same function for all images.
        if reference_images:
            for key in self.reference_images:
                getattr(self.reference_images[key], function_name)()

    def load_full_meta(self, reference_images=False):
        # Read the result file again.
        self.data = ImageProcessingData(self.folder, adjust_date=self.adjust_date)
        self.reference_images_iterator('load_full_meta', reference_images)

    def load_full_memmap(self, reference_images=False):
        # Read the result file again.
        self.load_full_meta()
        self.data.load_memmap_files()
        self.reference_images_iterator('load_full_memmap', reference_images)

    def remove_full_memmap(self, reference_images=False):
        # Remove all memmap files
        self.data.remove_memmap_files()
        self.reference_images_iterator('remove_full_memmap', reference_images)

    def load_slice_meta(self, reference_images=False):
        # Read the information for individual slices
        self.slice_json_paths = []
        for slice_folder, slice_name in zip(self.slice_folders, self.slice_names):
            self.slice_json_paths.append(os.path.join(slice_folder, 'info.json'))
            self.slice_data[slice_name] = ImageProcessingData(slice_folder, adjust_date=self.adjust_date)

        self.reference_images_iterator('load_slice_meta', reference_images)

    def load_slice_memmap(self, reference_images=False):
        # Read slices including memmap files
        self.load_slice_meta()
        for slice_name in self.slice_names:
            self.slice_data[slice_name].load_memmap_files()

        self.reference_images_iterator('load_slice_memmap', reference_images)

    def remove_slice_memmap(self, reference_images=False):
        # Remove memmap information
        for slice_name in self.slice_names:
            self.slice_data[slice_name].remove_memmap_files()

        self.reference_images_iterator('remove_slice_memmap', reference_images)

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
            elif 'coreg' in readfile_names and 'coreg' in orbit_prefix:
                coor.load_readfile(self.slice_data[slice].readfiles['coreg'])
                coor.load_orbit(self.slice_data[slice].orbits[orbit_names[orbit_prefix.index('coreg')]])
            else:
                coor.load_readfile(self.slice_data[slice].readfiles[readfile_names[0]])
                coor.load_orbit(self.slice_data[slice].orbits[orbit_names[0]])

            coors.append(coor)

        concat = CoorConcatenate(coors, adjust_date=self.adjust_date)
        concat.update_readfiles(coor.readfile, coor.orbit)

        concat_image = ImageProcessingData('')
        concat_image.meta.add_readfile(concat.readfile)
        concat_image.meta.add_orbit(coor.orbit)
        concat_image.meta.create_header()
        concat_image.path = os.path.join(os.path.dirname(self.slice_data[slice].folder), 'info.json')
        concat_image.save_json(concat_image.path)

        self.data = concat_image

    def create_concatenate_image(self, process, file_type, coor, data_id='', polarisation='', overwrite=False,
                                 output_type='disk', transition_type='full_weight', replace=False, cut_off=10,
                                 remove_input=False):
        """
        This method is used to concatenate slices. Be sure that before this step is run the meta data is first created
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
        succes = concatenate.create_image()
        # Do the actual concatenation
        if succes:
            concatenate.concatenate(transition_type=transition_type, cut_off=cut_off, replace=replace, remove_input=remove_input)
