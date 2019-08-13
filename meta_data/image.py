'''
This is a parent class for different types of images like SLC and interferogram
'''
import os

from rippl.meta_data.image_data import ImageData
from rippl.processing_steps.concatenate import Concatenate

class Image(object):

    """
    :type data = ImageData
    :type slice_list = list
    """

    def __init__(self, folder, slice_list='', update_full_image=False):
        # Either give an xml_file or a res_file as input to define the meta_data of the image
        # Every image will contain slices (or bursts). This could be one or multiple.
        # Processing will be done based on these slices. Although some processes for which not everything needs to be
        # read in memory can be done seperately.

        # Read the image information
        self.folder = folder
        self.meta_path = os.path.join(folder, 'info.json')
        self.data = ImageData(self.meta_path)

        # Other images that are linked to this image (added in SLC or interferogram)
        self.reference_images = dict()

        # Create the list of burst of this image
        self.slice_data = dict()

        # Read the individual slices ('slice_' + str(number) + '_swath_' + str(swath_no) + '_' + pol)
        # Length of these folders should be 20 characters
        self.slice_folders = next(os.walk(folder))[1]
        self.slice_names = sorted([x for x in self.slice_folders if len(x) == 20])

        # If a slice list is defined remove the slices which are not included.
        if slice_list:
            self.slice_names = list(set(self.slice_names) & set(slice_list))
        self.slice_folders = [os.path.join(folder, x) for x in self.slice_names]

        if not os.path.exists(self.meta_path) or update_full_image:
            self.load_slice_meta()
            concat = Concatenate.create_concat_meta([self.slice_data[key] for key in self.slice_data.keys()])
            concat.write(self.meta_path)
            self.slice_data = dict()

        self.load_full_meta()
        self.load_full_memmap()

    def reference_images_iterator(self, function_name, reference_images=True):
        # Run the same function for all images.
        if reference_images:
            for key in self.reference_images:
                getattr(self.reference_images[key], function_name)()

    def load_full_meta(self, reference_images=False):
        # Read the result file again.
        self.data = ImageData(self.meta_path)
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
        self.slice_meta_paths = []
        for slice_folder, slice_name in zip(self.slice_folders, self.slice_names):
            self.slice_meta_paths.append(os.path.join(slice_folder, 'info.json'))
            self.slice_data[slice_name] = ImageData(os.path.join(slice_folder, 'info.json'))

        self.reference_images_iterator('load_slice_meta', reference_images)

    def load_slice_memmap(self, reference_images=False):
        # Read slices including memmap files
        self.load_slice_meta()
        for slice_name in self.slice_names:
            self.slice_data[slice_name].read_data_memmap()

        self.reference_images_iterator('load_slice_memmap', reference_images)

    def remove_slice_memmap(self, reference_images=False):
        # Remove memmap information
        for slice_name in self.slice_names:
            self.slice_data[slice_name].remove_memmap_files()

        self.reference_images_iterator('remove_slice_memmap', reference_images)