'''
This is a parent class for different types of images like SLC and interferogram
'''
import os
from typing import List

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

    def __init__(self, folder, slice_list='', update_full_image=False, json_path=''):
        # Either give an xml_file or a res_file as input to define the meta_data of the image
        # Every image will contain slices (or bursts). This could be one or multiple.
        # Processing will be done based on these slices. Although some processes for which not everything needs to be
        # read in memory can be done seperately.

        # Read the image information
        self.folder = folder
        if not json_path:
            self.json_path = os.path.join(self.folder, 'info.json')
        else:
            self.json_path = json_path
        self.data = ImageProcessingData(self.folder)

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
            self.create_concatenate()
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
        # type: (ImageConcatData, List(str), List(CoordinateSystem), List(str), List(str), List(str), bool, bool, bool) -> (List(str), List(str), List(CoordinateSystem), List(str), List(ImageData))

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
                process_ids_out += process_ids_full
                file_types_out += file_types_full
                coordinates_out += coordinates_slice
                in_coordinates_out += in_coordinates_slice
                processes_out += processes_slice
                slice_names_out += [slice_name for n in range(len(process_ids_full))]
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
        self.data = ImageProcessingData(self.folder)
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
            self.slice_data[slice_name] = ImageProcessingData(slice_folder)

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

    def create_concatenate(self):
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

        concat = CoorConcatenate(coors)
        concat.update_readfiles(coor.readfile, coor.orbit)

        concat_image = ImageProcessingData('')
        concat_image.meta.add_readfile(concat.readfile)
        concat_image.meta.add_orbit(coor.orbit)
        concat_image.meta.create_header()
        concat_image.path = os.path.join(os.path.dirname(self.slice_data[slice].folder), 'info.json')
        concat_image.update_json(concat_image.path)

        self.data = concat_image

    def concatenate(self, process, process_type, coor, data_id='', polarisation='', overwrite=False, concat_output='disk'):
        # Create a concatenated image for a specific

        # First find the corresponding processes for the slices. If one is missing, throw an error.
        slice_names, process_ids, coordinates, file_types, images = self.concat_image_data_iterator(
            [process], [coor], process_types=[process_type], data_ids=[data_id], polarisations=[polarisation], slices=True, full_image=False)

        data_ids = [ProcessMeta.split_process_id(process_id)[3] for process_id in process_ids]
        polarisations = [ProcessMeta.split_process_id(process_id)[4] for process_id in process_ids]
        process_name = ProcessMeta.split_process_id(process_ids[0])[0]
        process_id = process_ids[0]

        if len(slice_names) == 0:
            print('No slices found to concatenate!')
            return
        if len(coordinates) != len(self.slice_names):
            print('Number of slices and found input data images is not the same. Aborting concatenation')
            return
        if not len(set(data_ids)) == 1 or not len(set(polarisations)) == 1:
            print('Please specify polarisation and/or data id to select the right process')
            return

        data_id = data_ids[0]
        polarisation = polarisations[0]

        # Then create the new concatenated coordinate system
        concat = CoorConcatenate(coordinates)
        concat_coor = concat.concat_coor

        # Based on the new coordinate system create a new process.
        # Check if process already exists in
        process_id_full = self.concat_image_data_iterator([process], [coor], process_types=[process_type],
                                                          data_ids=[data_id], polarisations=[polarisation], slices=False, full_image=True)
        if len(process_id_full) > 0 and overwrite == False:
            print('Concatenated dataset already exists. If you want to overwrite set overwrite to True')
            return
        elif self.data.process_id_exist(process_id_full):
            process = self.data.processes_data[process_name][process_id]
        else:
            process = ProcessData(process_name, concat_coor, data_id=data_id, polarisation=polarisation)

        # Create an empty image for the processing
        process.add_process_images([process_type], [images[0].dtype])
        full_image = self.concat_image_data_iterator([process], [coor], process_types=[process_type], data_ids=[data_id],
                                                polarisations=[polarisation], slices=False, full_image=True)[4][0]

        if concat_output == 'memory':
            full_image.new_memory_data(full_image.shape)     # type: ImageData
        else:
            full_image.create_disk_data()

        # Check if the full images are loaded in memory, otherwise use the data on disk to move to the new image.
        # If either of them is missing throw an error.

        for image, sync_coor in zip(images, concat.sync_coors):

            lin_0 = sync_coor.first_line - process.coordinates.first_line
            pix_0 = sync_coor.first_pixel - process.coordinates.first_pixel
            lin_size = sync_coor.shape[0]
            pix_size = sync_coor.shape[1]

            if image.data_memory_meta['shape'] == image.shape:
                if concat_output == 'memory':
                    full_image.data_memory[lin_0:lin_0 + lin_size, pix_0:pix_0 + pix_size] = image.data_memory
                else:
                    full_image.data_disk[lin_0:lin_0 + lin_size, pix_0:pix_0 + pix_size] = image.memory2disk(
                        image.data_memory, image.dtype)
            elif image.load_disk_data():
                if concat_output == 'memory':
                    full_image.data_memory[lin_0:lin_0 + lin_size, pix_0:pix_0 + pix_size] = image.disk2memory(
                        image.data_disk, image.dtype)
                else:
                    full_image.data_disk[lin_0:lin_0 + lin_size, pix_0:pix_0 + pix_size] = image.data_disk
